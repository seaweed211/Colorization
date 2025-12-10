import torch
import torch.nn as nn
from torchvision import transforms
import clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CLIPGuidance:
    def __init__(self):
        print(">>> [Guidance] Loading CLIP model...")
        self.model, _ = clip.load("ViT-B/32", device=device)
        self.norm = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                         (0.26862954, 0.26130258, 0.27577711))

    def get_best_prompts(self, image_tensor, masks, candidate_dict):
        # 1. 准备结果字典
        result_prompts = {}

        # 2. 遍历每个区域 (skin, hair, bg)
        for region, candidates in candidate_dict.items():
            mask = masks[region]
        
            # 如果 mask 太小（比如没头发），直接用全图，防止报错
            if mask.sum() < 10: 
                masked_img = image_tensor
            else:
                masked_img = image_tensor * mask
        
            # 预处理图片 (224x224)
            img_in = nn.functional.interpolate(masked_img, size=(224, 224), mode='bicubic')
            img_in = self.norm(img_in) # 使用类里初始化好的 norm
        
            # 预处理文本
            text_inputs = clip.tokenize(candidates).to(device)
        
            # 计算相似度
            with torch.no_grad():
                image_features = self.model.encode_image(img_in)
                text_features = self.model.encode_text(text_inputs)
            
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
            
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                best_idx = similarity.argmax().item()
            
            # 记录最佳词
            result_prompts[region] = candidates[best_idx]
        
        return result_prompts


    # def calc_loss(self, img_tensor, masks, text_tokens):  <--- 这是你之前的错误写法
    def calc_loss(self, img_tensor, masks, prompts_dict): # <--- 修正：变量名必须对应
        loss = 0
        device = img_tensor.device
        
        # 统一缩放原图
        raw_in = nn.functional.interpolate(img_tensor, (224, 224), mode='bicubic')
        
        for region in ['skin', 'hair', 'bg']:
            # 检查1：prompts_dict 里有没有这个区域的词？
            # 检查2：mask 是否有效？
            if region in prompts_dict and masks[region].sum() > 0:
                
                # --- A. 准备 Mask ---
                m = nn.functional.interpolate(masks[region], (224, 224), mode='nearest')
                
                # --- B. 准备图像输入 ---
                region_in = self.norm(raw_in * m)
                
                # --- C. 计算图像特征 (Image Embedding) ---
                img_emb = self.model.encode_image(region_in)
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
                
                # --- D. 处理文本 (Token化 + Text Embedding) ---
                # 这里的 prompts_dict 就是你从外面传进来的 {'skin': 'blue skin', ...}
                text_str = prompts_dict[region]
                text_token = clip.tokenize([text_str]).to(device)
                
                # 编码文本特征 (detach 防止梯度回传到 CLIP 文本塔)
                txt_emb = self.model.encode_text(text_token).detach()
                txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
                
                # --- E. 计算余弦距离损失 ---
                similarity = torch.cosine_similarity(img_emb, txt_emb)
                loss += (1 - similarity).mean()
                
        return loss