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