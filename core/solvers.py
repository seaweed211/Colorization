import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- VGG 相关组件 ---
class Normalization(nn.Module):
    def __init__(self):
        super().__init__()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)
        self.mean, self.std = mean, std
    def forward(self, img): return (img - self.mean) / self.std

class VGGFeatures(nn.Module):
    def __init__(self):
        super().__init__()
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
        for s in self.slices: x = s(x); results.append(x)
        return results

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    return torch.mm(features, features.t())

def get_tv_loss(img):
    """
    计算 Total Variation Loss，用于平滑图像，去噪点
    """
    b, c, h, w = img.size()
    h_tv = torch.pow((img[:, :, 1:, :] - img[:, :, :h-1, :]), 2).sum()
    w_tv = torch.pow((img[:, :, :, 1:] - img[:, :, :, :w-1]), 2).sum()
    return (h_tv + w_tv) / (b * c * h * w)

# --- 求解器基类 ---
class BaseSolver:
    def run(self, init_img, style_img, c_masks, s_masks, clip_selector, prompts):
        raise NotImplementedError

class OptimizationSolver(BaseSolver):
    # def __init__(self, iterations=500, content_weight=1.0, clip_weight=10.0, style_weight=0.0, tv_weight=1e-3):
    def __init__(self, iterations=500, content_weight=150, clip_weight=0.0, style_weight=0.0, tv_weight=1e-3):
        self.iterations = iterations
        self.content_weight = content_weight # 必须有，保证结构
        self.clip_weight = clip_weight       # 核心驱动力
        self.style_weight = style_weight     # 可选：如果还需要一点点纹理迁移，可以设为非0
        self.tv_weight = tv_weight
        
        self.vgg = VGGFeatures().to(device)
        self.norm = Normalization().to(device)

    def run(self, init_img, style_img, c_masks, s_masks, clip_selector, target_features):
        print(f">>> [Solver] 开始优化 (使用 Adam)...")
        
        # 1. 准备目标特征
        with torch.no_grad():
            content_feats = self.vgg(self.norm(init_img))
            content_target = content_feats[3].detach() # conv4_2

        # 2. 设置优化变量
        opt_img = init_img.clone().detach().requires_grad_(True)
        
        # 【核心修改】使用 Adam，学习率设为 0.02
        optimizer = optim.Adam([opt_img], lr=0.02)
        
        # 3. 迭代循环 (Adam 不需要 closure)
        import tqdm
        pbar = tqdm.tqdm(range(self.iterations)) # 建议 iterations 设为 500-1000
        
        for i in pbar:
            optimizer.zero_grad()
            
            # --- 强制约束像素在 [0, 1] 范围内 ---
            # 这一步非常重要，防止像素值溢出导致 NaN
            opt_img.data.clamp_(0, 1)
            
            # --- A. Content Loss ---
            curr_feats = self.vgg(self.norm(opt_img))
            loss_content = nn.functional.mse_loss(curr_feats[3], content_target)
            
            # --- B. CLIP Loss ---
            loss_clip = torch.tensor(0.0).to(device)
            if clip_selector and target_features:
                loss_clip = clip_selector.calc_loss(opt_img, c_masks, target_features)
                
            # --- C. TV Loss (防止噪点) ---
            # 如果你还没有定义 get_tv_loss，记得加上，或者暂时注释掉
            loss_tv = torch.tensor(0.0).to(device) 
            # loss_tv = get_tv_loss(opt_img) 

            # --- D. 总 Loss ---
            total_loss = (loss_content * self.content_weight) + \
                         (loss_clip * self.clip_weight) + \
                         (loss_tv * self.tv_weight)
            
            # --- E. 反向传播 ---
            total_loss.backward()
            
            # 【核心修改】梯度裁剪 (Gradient Clipping)
            # 这行代码是防止 NaN 的最后一道防线
            torch.nn.utils.clip_grad_norm_([opt_img], max_norm=1.0)
            
            optimizer.step()
        
        # 最后一次 clamp 确保输出合法
        with torch.no_grad():
            opt_img.clamp_(0, 1)
            
        return opt_img