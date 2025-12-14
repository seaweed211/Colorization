import torch
import torch.nn as nn
from torchvision import transforms
import clip

class CLIPGuidance:
    def __init__(self, device):
        self.device = device
        print(f">>> [Guidance] Loading CLIP model on {device}...")
        self.model, _ = clip.load("ViT-B/32", device=device)
        # CLIP 标准预处理参数
        self.norm = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                         (0.26862954, 0.26130258, 0.27577711))

    def get_image_embeddings(self, image_tensor, masks):
        """
        [新功能] 预先提取参考图(Style Image)各个区域的 CLIP 特征
        这步只做一次，存下来给后面反复用。
        """
        ref_features = {}
        
        # 统一缩放到 CLIP 需要的尺寸
        raw_in = nn.functional.interpolate(image_tensor, (224, 224), mode='bicubic')

        with torch.no_grad():
            for region, mask in masks.items():
                # 忽略空的区域（比如参考图里没有头发，就不提取头发特征）
                if mask.sum() < 100: 
                    continue

                # 1. 缩放 mask
                m = nn.functional.interpolate(mask, (224, 224), mode='nearest')
                
                # 2. 抠图 + 归一化
                # 注意：这里我们让背景变黑(0)，只保留该区域像素
                region_in = self.norm(raw_in * m)
                
                # 3. 提取特征
                emb = self.model.encode_image(region_in)
                emb = emb / emb.norm(dim=-1, keepdim=True) # 归一化特征向量
                
                ref_features[region] = emb
        
        return ref_features

    def calc_loss(self, img_tensor, masks, ref_features):
        """
        计算当前生成图与参考图特征的距离
        img_tensor: 当前生成的图
        masks: 当前输入图的 masks (Input Masks)
        ref_features: 上面算好的参考图特征字典
        """
        loss = 0
        count = 0
        
        # 统一缩放
        raw_in = nn.functional.interpolate(img_tensor, (224, 224), mode='bicubic')
        
        # 遍历参考图里有的特征（交集逻辑）
        for region, target_emb in ref_features.items():
            
            # 【关键逻辑】只有当输入图也有这个区域时，才计算 Loss
            # 例子：参考图有头发(ref_features里有'hair')，但输入图是光头(masks['hair']为空) -> 跳过
            if region in masks and masks[region].sum() > 100:
                
                # 1. 准备当前图片的 Mask 区域
                m = nn.functional.interpolate(masks[region], (224, 224), mode='nearest')
                region_in = self.norm(raw_in * m)
                
                # 2. 提取当前生成的特征
                current_emb = self.model.encode_image(region_in)
                current_emb = current_emb / current_emb.norm(dim=-1, keepdim=True)
                
                # 3. 计算 Cosine Loss (越相似，Similarity越大，Loss越小)
                # target_emb 必须 detach，虽然从 get_image_embeddings 出来默认就是 detached，但保险起见
                similarity = torch.cosine_similarity(current_emb, target_emb.detach())
                
                loss += (1.0 - similarity).mean()
                count += 1
                
        if count > 0:
            return loss / count
        else:
            return torch.tensor(0.0, device=img_tensor.device, requires_grad=True)