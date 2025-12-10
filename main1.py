import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import clip
import numpy as np
import cv2
import os
import copy
from mask_utils import get_semantic_masks

# ==================== 1. 配置区域 ====================
CONTENT_IMG_PATH = "inputs/in1.jpg"   
STYLE_IMG_PATH = "styles/ref1.jpg"     
OUTPUT_PATH = "outputs/out1.jpg"
IMG_SIZE = 512 

# --- 权重策略 (配合新的 Gram 计算方式已调整) ---
CONTENT_WEIGHT = 0.0     
STYLE_WEIGHT = 1e4       # <--- 降低了，因为 Gram 值变大了
CLIP_WEIGHT = 5000.0     
TV_WEIGHT = 10.0         
ITERATIONS = 500         

PROMPT_POOLS = {
    "skin": ["blue skin", "Avatar Na'vi skin", "dark blue skin", "bioluminescent skin"],
    "hair": ["black hair", "braided hair", "dark hair"],
    "bg": ["forest background", "jungle", "dark background", "nebula"]
}

# VGG 标准化参数
VGG_MEAN = torch.tensor([0.485, 0.456, 0.406])
VGG_STD = torch.tensor([0.229, 0.224, 0.225])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# ==================== 2. 核心工具函数 ====================

def load_image_with_ratio(image_name, size=512):
    image = Image.open(image_name).convert("RGB")
    w, h = image.size
    ratio = size / max(w, h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    image = image.resize((new_w, new_h), Image.BICUBIC)
    transform = transforms.Compose([transforms.ToTensor()])
    tensor = transform(image).unsqueeze(0).to(device, torch.float)
    return tensor, new_w, new_h

def gram_matrix(input):
    """
    修改版：移除除法！让 Loss 数值变大，避免梯度消失。
    """
    a, b, c, d = input.size() 
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    # return G.div(a * b * c * d) # <--- 删掉了除法
    return G # 直接返回巨大的数值

def calc_mean_std(feat, eps=1e-5):
    """计算特征的均值和标准差"""
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def color_transfer_region_aware(content_img, style_img, content_masks, ref_masks):
    """
    【核心大招】区域感知颜色迁移初始化
    在 VGG 跑之前，先把 Style 的颜色“硬贴”到 Content 上。
    """
    print(">>> 正在执行区域感知颜色初始化 (Region-Aware Color Transfer)...")
    res_img = content_img.clone()
    
    for region in ['skin', 'hair', 'bg']:
        # 获取 Mask
        c_mask = content_masks[region]
        s_mask = ref_masks[region]
        
        # 只有当两边都有 Mask 时才迁移
        if c_mask.sum() > 10 and s_mask.sum() > 10:
            # 1. 提取该区域的像素
            c_vals = torch.masked_select(content_img, c_mask.bool()).view(3, -1)
            s_vals = torch.masked_select(style_img, s_mask.bool()).view(3, -1)
            
            # 2. 计算均值方差
            mu_c, std_c = c_vals.mean(1, keepdim=True), c_vals.std(1, keepdim=True)
            mu_s, std_s = s_vals.mean(1, keepdim=True), s_vals.std(1, keepdim=True)
            
            # 3. 颜色统计对齐 (AdaIN 逻辑)
            # (x - mu_c) * (std_s / std_c) + mu_s
            transferred_vals = (c_vals - mu_c) / (std_c + 1e-6) * (std_s + 1e-6) + mu_s
            
            # 4. 填回原图 (需要复杂的索引操作，这里用 Mask 逐像素填充)
            # 为了简单高效，我们生成一张全图的 transferred_layer
            # 这种写法在 Mask 复杂时可能稍慢，但逻辑最清晰
            
            # 简易版：针对全图做一次对齐，然后只取 Mask 部分
            # 注意：这只是为了初始化，不需要太完美，只要颜色偏向正确即可
            
            # 更精确的做法：
            # 我们直接把 res_img 对应区域的值改掉
            # 由于 masked_select 展平了 tensor，我们需要用 masked_scatter_
            res_img.masked_scatter_(c_mask.bool(), transferred_vals)
            print(f"    -> 已初始化区域: {region} (颜色已迁移)")
        else:
            print(f"    -> 跳过区域 {region} (Mask 缺失)")
            
    return res_img

def transfer_color_only(content_tensor, styled_tensor):
    """YUV 亮度融合"""
    def to_numpy(tensor):
        img = tensor.detach().cpu().clone().squeeze(0).permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        return (img * 255).astype(np.uint8)

    content_img = to_numpy(content_tensor)
    styled_img = to_numpy(styled_tensor)
    content_yuv = cv2.cvtColor(content_img, cv2.COLOR_RGB2YUV)
    styled_yuv = cv2.cvtColor(styled_img, cv2.COLOR_RGB2YUV)
    final_yuv = styled_yuv.copy()
    final_yuv[:, :, 0] = content_yuv[:, :, 0] 
    final_rgb = cv2.cvtColor(final_yuv, cv2.COLOR_YUV2RGB)
    loader = transforms.Compose([transforms.ToTensor()])
    return loader(Image.fromarray(final_rgb)).unsqueeze(0).to(device)

# ==================== 3. 类定义 ====================

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1).to(device)
        self.std = std.clone().detach().view(-1, 1, 1).to(device)
    def forward(self, img):
        return (img - self.mean) / self.std

class CLIP_Selector:
    def __init__(self, device):
        self.model, _ = clip.load("ViT-B/32", device=device)
        self.device = device  
    def pick_best_prompt(self, image_tensor, mask_tensor, candidates):
        if mask_tensor.sum() < 10: masked_img = image_tensor
        else: masked_img = image_tensor * mask_tensor
        img_in = nn.functional.interpolate(masked_img, size=(224, 224), mode='bicubic')
        norm = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                    (0.26862954, 0.26130258, 0.27577711))
        img_in = norm(img_in)
        text_inputs = clip.tokenize(candidates).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(img_in)
            text_features = self.model.encode_text(text_inputs)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            best_idx = similarity.argmax().item()
        return candidates[best_idx]

class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        vgg = models.vgg19(pretrained=True).features.eval()
        self.slices = nn.ModuleList([
            nn.Sequential(*list(vgg.children())[:2]),   
            nn.Sequential(*list(vgg.children())[2:7]),  
            nn.Sequential(*list(vgg.children())[7:12]), 
            nn.Sequential(*list(vgg.children())[12:21]),
            nn.Sequential(*list(vgg.children())[21:30]) 
        ])
    def forward(self, x):
        results = []
        for i, slice in enumerate(self.slices):
            x = slice(x)
            results.append(x)
        return results

# ==================== 4. 主程序 ====================

def main():
    print("1. 加载资源...")
    content_img, cw, ch = load_image_with_ratio(CONTENT_IMG_PATH, IMG_SIZE)
    style_loader = transforms.Compose([transforms.Resize((ch, cw)), transforms.ToTensor()])
    style_img = style_loader(Image.open(STYLE_IMG_PATH).convert("RGB")).unsqueeze(0).to(device)
    
    normalization = Normalization(VGG_MEAN, VGG_STD).to(device)
    vgg = VGGFeatures().to(device)
    clip_model, _ = clip.load("ViT-B/32", device=device)
    
    print("2. 生成 Mask...")
    content_masks = get_semantic_masks(CONTENT_IMG_PATH, cw, ch)
    for k in content_masks: content_masks[k] = content_masks[k].to(device)
    ref_masks = get_semantic_masks(STYLE_IMG_PATH, cw, ch)
    for k in ref_masks: ref_masks[k] = ref_masks[k].to(device)

    # 兜底：如果参考图Mask为空，全图填充
    if ref_masks['skin'].sum() == 0:
        print("!!! 警告: 参考图 Mask 空，使用全图兜底 !!!")
        ref_masks['skin'] = torch.ones_like(ref_masks['skin'])

    # --- 关键步骤：颜色初始化 ---
    # 这将确保优化器从有颜色的图片开始跑，而不是从灰度图开始
    init_img = color_transfer_region_aware(content_img, style_img, content_masks, ref_masks)
    
    print("3. CLIP 分析...")
    selector = CLIP_Selector(device)
    prompts = {}
    for region in ['skin', 'hair', 'bg']:
        prompts[region] = selector.pick_best_prompt(style_img, ref_masks[region], PROMPT_POOLS[region])
    print(f"--> 分析结果: {prompts}")
    text_tokens = {k: clip.tokenize([v]).to(device) for k, v in prompts.items()}
    
    print("4. 计算 Target Gram (Global + Region)...")
    style_targets = {}
    with torch.no_grad():
        norm_style_img = normalization(style_img)
        style_features = vgg(norm_style_img)
        for layer_idx, feat in enumerate(style_features):
            b, c, h, w = feat.shape
            # 全局兜底 Target
            style_targets[f"global_{layer_idx}"] = gram_matrix(feat)
            
            for region in ['skin', 'hair', 'bg']:
                mask_resized = nn.functional.interpolate(ref_masks[region], size=(h, w), mode='nearest')
                if mask_resized.sum() > 10:
                    style_targets[f"{region}_{layer_idx}"] = gram_matrix(feat * mask_resized)
                else:
                    # 如果参考图缺区域，用全局兜底
                    style_targets[f"{region}_{layer_idx}"] = style_targets[f"global_{layer_idx}"]

    print("5. 开始优化 (Initialized with Color)...")
    # 重点：使用已经上好色的 init_img 开始优化！
    opt_img = init_img.clone().requires_grad_(True)
    optimizer = optim.LBFGS([opt_img])
    
    run = [0]
    while run[0] <= ITERATIONS:
        def closure():
            with torch.no_grad(): opt_img.clamp_(0, 1)
            optimizer.zero_grad()
            
            norm_opt_img = normalization(opt_img)
            current_features = vgg(norm_opt_img)
            
            loss_style = 0
            loss_clip = 0
            loss_tv = 0
            
            # Style Loss
            for layer_idx, feat in enumerate(current_features):
                b, c, h, w = feat.shape
                for region in ['skin', 'hair', 'bg']:
                    if content_masks[region].sum() > 0:
                        target_key = f"{region}_{layer_idx}"
                        mask_resized = nn.functional.interpolate(content_masks[region], size=(h, w), mode='nearest')
                        current_gram = gram_matrix(feat * mask_resized)
                        # 直接与目标 Gram 计算 MSE
                        loss_style += nn.functional.mse_loss(current_gram, style_targets[target_key])
            
            # CLIP Loss
            clip_input_raw = nn.functional.interpolate(opt_img, size=(224, 224), mode='bicubic')
            clip_norm = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                             (0.26862954, 0.26130258, 0.27577711))
            for region in ['skin', 'hair', 'bg']:
                if content_masks[region].sum() > 0:
                    mask_for_clip = nn.functional.interpolate(content_masks[region], size=(224, 224), mode='nearest')
                    region_input = clip_norm(clip_input_raw * mask_for_clip)
                    img_embed = clip_model.encode_image(region_input)
                    text_embed = clip_model.encode_text(text_tokens[region])
                    loss_clip += (1 - torch.cosine_similarity(img_embed, text_embed)).mean()

            loss_tv = torch.sum(torch.abs(opt_img[:, :, :, :-1] - opt_img[:, :, :, 1:])) + \
                      torch.sum(torch.abs(opt_img[:, :, :-1, :] - opt_img[:, :, 1:, :]))

            total_loss = (loss_style * STYLE_WEIGHT) + (loss_clip * CLIP_WEIGHT) + (loss_tv * TV_WEIGHT)
            total_loss.backward()
            run[0] += 1
            if run[0] % 50 == 0:
                # 现在的 Loss 会很大，这是正常的
                print(f"Step {run[0]}: Style: {loss_style.item():.2f}, CLIP: {loss_clip.item():.4f}")
            return total_loss
            
        optimizer.step(closure)
    
    print("6. YUV 融合保存...")
    # 注意：这里我们融合回 Content Img 的亮度，但保留生成的色度
    final_output = transfer_color_only(content_img, opt_img)
    if not os.path.exists("outputs"): os.makedirs("outputs")
    save_img = final_output.detach().cpu().clone().squeeze(0)
    unloader = transforms.ToPILImage()
    image = unloader(save_img)
    image.save(OUTPUT_PATH)
    print("完成！")

if __name__ == "__main__":
    main()