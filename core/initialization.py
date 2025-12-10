import torch

class ColorInitializer:
    def process(self, content_img, style_img, c_masks, s_masks):
        raise NotImplementedError

class AdaINRegionAware(ColorInitializer):
    """基于区域统计量的 AdaIN 初始化"""
    def process(self, content_img, style_img, c_masks, s_masks):
        print(">>> [Init] 执行 AdaIN 区域颜色移植...")
        res = content_img.clone()
        for region in ['skin', 'hair', 'bg']:
            c_mask, s_mask = c_masks[region], s_masks[region]
            if c_mask.sum() > 10 and s_mask.sum() > 10:
                # 提取像素
                c_vals = torch.masked_select(content_img, c_mask.bool()).view(3, -1)
                s_vals = torch.masked_select(style_img, s_mask.bool()).view(3, -1)
                # 计算统计量
                mu_c, std_c = c_vals.mean(1, keepdim=True), c_vals.std(1, keepdim=True)
                mu_s, std_s = s_vals.mean(1, keepdim=True), s_vals.std(1, keepdim=True)
                # 对齐
                transferred = (c_vals - mu_c) / (std_c + 1e-6) * (std_s + 1e-6) + mu_s
                # 填回
                res.masked_scatter_(c_mask.bool(), transferred)
        return res

class GrayStart(ColorInitializer):
    """不做任何初始化，从灰度开始"""
    def process(self, content_img, *args):
        print(">>> [Init] 保持灰度原图...")
        return content_img.clone()