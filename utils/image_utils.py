import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image_with_ratio(image_name, size=512):
    image = Image.open(image_name).convert("RGB")
    w, h = image.size
    ratio = size / max(w, h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    image = image.resize((new_w, new_h), Image.BICUBIC)
    transform = transforms.Compose([transforms.ToTensor()])
    tensor = transform(image).unsqueeze(0).to(device, torch.float)
    return tensor, new_w, new_h

def save_result(tensor, path):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone().squeeze(0)
    image = unloader(image)
    image.save(path)

def transfer_color_presolve_luminance(content_tensor, styled_tensor):
    """YUV 亮度融合"""
    def to_numpy(t):
        img = t.detach().cpu().clone().squeeze(0).permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        return (img * 255).astype(np.uint8)

    content_img = to_numpy(content_tensor)
    styled_img = to_numpy(styled_tensor)
    
    c_yuv = cv2.cvtColor(content_img, cv2.COLOR_RGB2YUV)
    s_yuv = cv2.cvtColor(styled_img, cv2.COLOR_RGB2YUV)
    
    final_yuv = s_yuv.copy()
    final_yuv[:, :, 0] = c_yuv[:, :, 0] # 强制使用原图亮度
    
    final_rgb = cv2.cvtColor(final_yuv, cv2.COLOR_YUV2RGB)
    loader = transforms.Compose([transforms.ToTensor()])
    return loader(Image.fromarray(final_rgb)).unsqueeze(0).to(device)



def smart_color_merge(content_tensor, result_tensor):
    """
    终极版融合：高频细节注入法 (Frequency Separation)
    1. 基底 (Base): 完全信任 Result 的颜色、光影和色调。
    2. 细节 (Detail): 从 Content 中提取纯粹的纹理（去除了颜色和光影）。
    3. 融合: Base + Detail。
    这样可以保证颜色 100% 还原 final_raw，同时拥有 content 的清晰度。
    """
    
    # --- 辅助：转 Numpy ---
    def to_numpy(t):
        if t.dim() == 4: t = t.squeeze(0)
        img = t.clone().detach().cpu().permute(1, 2, 0).numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 1. 准备数据
    img_content = to_numpy(content_tensor)
    img_result = to_numpy(result_tensor)

    # 2. 转到 LAB 空间
    # content 提供细节，result 提供颜色和基调
    lab_content = cv2.cvtColor(img_content, cv2.COLOR_BGR2LAB)
    lab_result = cv2.cvtColor(img_result, cv2.COLOR_BGR2LAB)

    l_c, _, _ = cv2.split(lab_content)
    l_r, a_r, b_r = cv2.split(lab_result)

    # --- 步骤 A: 处理 Result (基底) ---
    # Result 虽然颜色对，但有噪点。我们需要一个“干净的颜色基底”。
    # 对 l_r, a_r, b_r 做双边滤波，保留主要边缘（轮廓），去掉高频噪点
    # d=9, sigmaColor=75, sigmaSpace=75
    l_r_smooth = cv2.bilateralFilter(l_r, 9, 75, 75)
    a_r_smooth = cv2.bilateralFilter(a_r, 9, 75, 75)
    b_r_smooth = cv2.bilateralFilter(b_r, 9, 75, 75)

    # --- 步骤 B: 提取 Content 的高频细节 (纹理) ---
    # 细节 = 原图 - 模糊后的原图
    # 这里用高斯模糊来模拟“低频光影”
    blur_size = 21 # 模糊核大小，控制提取细节的粗细
    l_c_blur = cv2.GaussianBlur(l_c, (blur_size, blur_size), 0)
    
    # 计算细节差值 (转成 int16 防止负数截断)
    # detail 包含了毛孔、发丝、眼睛轮廓，但不包含肤色明暗
    l_detail = l_c.astype(np.int16) - l_c_blur.astype(np.int16)

    # --- 步骤 C: 注入细节 ---
    # 新亮度 = Result的光影基底 + Content的纹理细节
    l_new = l_r_smooth.astype(np.int16) + l_detail
    
    # 稍微增强一点纹理的对比度 (可选，让画面更锐利)
    # l_new = l_r_smooth + l_detail * 1.2 

    # 限制回 0-255
    l_new = np.clip(l_new, 0, 255).astype(np.uint8)

    # --- 步骤 D: 合成 ---
    # 使用：注入纹理后的亮度 + Result原本的颜色(去噪后)
    # 这样完全保留了 Result 的色相(Hue)和饱和度(Saturation)
    lab_merged = cv2.merge([l_new, a_r_smooth, b_r_smooth])
    
    return cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)



def refine_masks(masks_dict, dilate_iter=3, blur_kernel=15):
    # """
    # 对 Mask 进行膨胀和羽化，消除锯齿和缝隙
    # masks_dict: 包含 'skin', 'hair' 等 tensor mask 的字典
    # dilate_iter: 膨胀力度，越大缝隙填补越强
    # blur_kernel: 羽化程度，必须是奇数，越大边缘越柔和
    # """
    refined_masks = {}
    
    # 定义膨胀核
    kernel = np.ones((3, 3), np.uint8)
    
    for k, v in masks_dict.items():
        # 1. Tensor -> Numpy (假设输入是 [1, 1, H, W] 或 [1, H, W])
        if isinstance(v, torch.Tensor):
            mask_np = v.squeeze().cpu().numpy().astype(np.float32)
        else:
            mask_np = v.astype(np.float32)
            
        # 2. 膨胀 (Dilation) - 核心步骤：消除缝隙/白边
        # 这一步会让 mask 向外生长，填补脸和头发中间的空隙
        if dilate_iter > 0:
            mask_np = cv2.dilate(mask_np, kernel, iterations=dilate_iter)
            
        # 3. 高斯模糊 (Gaussian Blur) - 核心步骤：消除锯齿
        # 这一步让边缘变成渐变
        if blur_kernel > 0:
            mask_np = cv2.GaussianBlur(mask_np, (blur_kernel, blur_kernel), 0)
            
        # 4. 重新归一化并转回 Tensor
        # 模糊后数值可能会变小，或者膨胀后重叠，这里稍微限制一下
        mask_np = np.clip(mask_np, 0, 1)
        
        # 转回 Tensor
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
        refined_masks[k] = mask_tensor
        
    return refined_masks