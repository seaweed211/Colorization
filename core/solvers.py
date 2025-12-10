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

# --- 求解器基类 ---
class BaseSolver:
    def run(self, init_img, style_img, c_masks, s_masks, clip_selector, prompts):
        raise NotImplementedError

# --- 具体的优化求解器 (VGG + LBFGS) ---
class OptimizationSolver(BaseSolver):
    def __init__(self, iterations=500, style_weight=1e4, clip_weight=5000):
        self.iterations = iterations
        self.style_weight = style_weight
        self.clip_weight = clip_weight
        self.vgg = VGGFeatures().to(device)
        self.norm = Normalization().to(device)

    def run(self, init_img, style_img, c_masks, s_masks, clip_selector, prompts):
        print(f">>> [Solver] 开始 VGG 迭代优化 ({self.iterations} steps)...")
        
        # 1. 预计算 Target Gram
        style_targets = {}
        with torch.no_grad():
            style_feats = self.vgg(self.norm(style_img))
            for i, feat in enumerate(style_feats):
                # 全局兜底
                style_targets[f"global_{i}"] = gram_matrix(feat)
                for region in ['skin', 'hair', 'bg']:
                    mask = nn.functional.interpolate(s_masks[region], size=feat.shape[2:], mode='nearest')
                    if mask.sum() > 10:
                        style_targets[f"{region}_{i}"] = gram_matrix(feat * mask)
                    else:
                        style_targets[f"{region}_{i}"] = style_targets[f"global_{i}"]

        # 2. 准备优化
        opt_img = init_img.clone().requires_grad_(True)
        optimizer = optim.LBFGS([opt_img])
        
        run = [0]
        while run[0] <= self.iterations:
            def closure():
                with torch.no_grad(): opt_img.clamp_(0, 1)
                optimizer.zero_grad()
                
                curr_feats = self.vgg(self.norm(opt_img))
                loss_style = 0
                loss_clip = 0
                
                # Style Loss
                for i, feat in enumerate(curr_feats):
                    for region in ['skin', 'hair', 'bg']:
                        if c_masks[region].sum() > 0:
                            mask = nn.functional.interpolate(c_masks[region], size=feat.shape[2:], mode='nearest')
                            curr_gram = gram_matrix(feat * mask)
                            loss_style += nn.functional.mse_loss(curr_gram, style_targets[f"{region}_{i}"])
                
                # CLIP Loss
                if clip_selector:
                    loss_clip = clip_selector.calc_loss(opt_img, c_masks, prompts)
                
                total = loss_style * self.style_weight + loss_clip * self.clip_weight
                total.backward()
                
                run[0] += 1
                if run[0] % 50 == 0:
                    s_val = loss_style.item() if hasattr(loss_style, 'item') else loss_style
                    c_val = loss_clip.item() if hasattr(loss_clip, 'item') else loss_clip

                    print(f"    Step {run[0]}: Style={s_val:.2f}, CLIP={c_val:.4f}")
                    # print(f"    Step {run[0]}: Style={loss_style.item():.2f}, CLIP={loss_clip.item():.4f}")
                return total
            
            optimizer.step(closure)
        return opt_img