import torch

class ColorInitializer:
    def process(self, content_img, style_img, c_masks, s_masks):
        raise NotImplementedError

class AdaINRegionAware(ColorInitializer):
    """基于区域统计量的 AdaIN 初始化"""
    def process(self, content_img, style_img, c_masks, s_masks):
        print(">>> [Init] 执行 AdaIN 区域颜色移植...")
        res = content_img.clone()
        
        # 支持的区域列表（根据实际存在的mask动态调整）
        regions = ['skin', 'hair', 'clothes', 'bg']
        
        for region in regions:
            # 跳过不存在的区域
            if region not in c_masks or region not in s_masks:
                continue
                
            c_mask, s_mask = c_masks[region], s_masks[region]
            if c_mask.sum() > 10 and s_mask.sum() > 10:
                # 【修复】将 mask 扩展到和 image 相同的形状
                # content_img: [1, 3, H, W], c_mask: [1, 1, H, W]
                # 需要扩展为 [1, 3, H, W] 才能正确使用 masked_select
                c_mask_expanded = c_mask.expand_as(content_img)
                s_mask_expanded = s_mask.expand_as(style_img)
                
                # 提取像素
                c_vals = torch.masked_select(content_img, c_mask_expanded.bool()).view(3, -1)
                s_vals = torch.masked_select(style_img, s_mask_expanded.bool()).view(3, -1)
                # 计算统计量
                mu_c, std_c = c_vals.mean(1, keepdim=True), c_vals.std(1, keepdim=True)
                mu_s, std_s = s_vals.mean(1, keepdim=True), s_vals.std(1, keepdim=True)
                # 对齐
                transferred = (c_vals - mu_c) / (std_c + 1e-6) * (std_s + 1e-6) + mu_s
                # 填回
                res.masked_scatter_(c_mask_expanded.bool(), transferred)
        return res

class GrayStart(ColorInitializer):
    """不做任何初始化，从灰度开始"""
    def process(self, content_img, *args):
        print(">>> [Init] 保持灰度原图...")
        return content_img.clone()