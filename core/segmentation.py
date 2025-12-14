import torch
import numpy as np
import cv2
import mediapipe as mp
import os

class BaseSegmenter:
    def get_masks(self, image_path, width, height):
        raise NotImplementedError

class MediaPipeSegmenter(BaseSegmenter):
    """原本的 MediaPipe 逻辑封装"""
    def get_masks(self, image_path, width, height):
        mp_face_mesh = mp.solutions.face_mesh
        mp_selfie = mp.solutions.selfie_segmentation
        
        # 预处理
        img = cv2.imread(image_path)
        if img is None: raise ValueError(f"无法读取: {image_path}")
        img = cv2.resize(img, (width, height))
        if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 1. Selfie Segmentation
        with mp_selfie.SelfieSegmentation(model_selection=1) as selfie:
            res = selfie.process(img_rgb)
            person_mask = (res.segmentation_mask > 0.1).astype(np.uint8) if res.segmentation_mask is not None else np.ones((height, width), np.uint8)

        # 2. Face Mesh
        face_mask = np.zeros((height, width), dtype=np.uint8)
        chin_y = None
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.1) as mesh:
            res = mesh.process(img_rgb)
            if res.multi_face_landmarks:
                landmarks = res.multi_face_landmarks[0].landmark
                points = [[int(l.x * width), int(l.y * height)] for l in landmarks]
                cv2.fillConvexPoly(face_mask, cv2.convexHull(np.array(points, np.int32)), 1)
                chin_y = int(landmarks[152].y * height)

        # 3. Logic
        final_bg = 1 - person_mask
        if chin_y:
            rem = np.clip(person_mask - face_mask, 0, 1)
            y_coords = np.arange(height).reshape(height, 1).repeat(width, axis=1)
            split = chin_y + int(height * 0.05)
            body = np.logical_and(rem==1, y_coords > split).astype(np.uint8)
            hair = np.logical_and(rem==1, y_coords <= split).astype(np.uint8)
            final_skin = np.clip(face_mask + body, 0, 1)
            final_hair = hair
        else:
            final_skin = person_mask
            final_hair = np.zeros_like(person_mask)

        # Helper
        def to_tensor(m): return torch.from_numpy(m).float().unsqueeze(0).unsqueeze(0)
        
        return {
            "skin": to_tensor(final_skin),
            "hair": to_tensor(final_hair),
            "bg": to_tensor(final_bg)
        }


class GlobalSegmenter(BaseSegmenter):
    """
    全局分割器：不进行任何分割，直接把整张图视为一个区域 (skin)。
    用于实现纯粹的全局风格迁移。
    """
    def get_masks(self, image_path, width, height):
        # 创建一个全白的 mask (全是1)
        full_mask = np.ones((height, width), dtype=np.uint8)
        # 创建一个全黑的 mask (全是0)
        empty_mask = np.zeros((height, width), dtype=np.uint8)
        
        def to_tensor(m): 
            return torch.from_numpy(m).float().unsqueeze(0).unsqueeze(0)
        
        # 我们把全图都归为 "skin" (或者你可以理解为 content)，其他区域为空
        # 这样 AdaIN 就会计算全图的均值方差，变成 Global AdaIN
        return {
            "skin": to_tensor(full_mask), 
            "hair": to_tensor(empty_mask),
            "bg":   to_tensor(empty_mask)
        }


class BiSeNetSegmenter(BaseSegmenter):
    """
    BiSeNet 人脸解析分割器：使用 BiSeNet 进行 19 类人脸解析，
    并将结果合并为 hair/skin/bg 三类 mask。
    """
    def __init__(self, model_path='core/79999_iter.pth'):
        """
        初始化 BiSeNet 模型
        Args:
            model_path: 模型权重文件路径
        """
        from .model import BiSeNet
        from collections import OrderedDict
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = BiSeNet(n_classes=19)
        self.net.to(self.device)
        
        # 加载模型权重
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            # 处理 DataParallel 的 module 前缀
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            self.net.load_state_dict(new_state_dict)
            self.net.eval()
            print(f"BiSeNet 模型加载成功: {model_path}")
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    def get_masks(self, image_path, width, height):
        """
        使用 BiSeNet 获取 hair/skin/bg 三类 mask
        
        BiSeNet 19 类定义:
        0:背景, 1:脸, 2-9:五官, 10:鼻, 11-13:嘴, 14-15:脖子, 16:衣, 17:发, 18:帽
        
        合并规则:
        - hair: class 17 (头发)
        - skin: class 1-15 (脸部、五官、脖子)
        - bg: class 0, 16, 18 (背景、衣服、帽子)
        """
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        # 转换为 RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # BiSeNet 需要 512x512 输入
        img_512 = cv2.resize(img_rgb, (512, 512))
        
        # 归一化
        img_tensor = torch.from_numpy(img_512).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # 推理
        with torch.no_grad():
            out = self.net(img_tensor)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
        
        # 还原到目标尺寸
        parsing = cv2.resize(parsing.astype(np.uint8), (width, height), 
                           interpolation=cv2.INTER_NEAREST)
        
        # 创建4类 mask
        # 头发
        hair_mask = (parsing == 17).astype(np.uint8)  
        
        # 皮肤：脸部 + 五官 + 脖子 (class 1-15)
        skin_indices = list(range(1, 16))
        skin_mask = np.isin(parsing, skin_indices).astype(np.uint8)

        # 衣服
        clothes_mask = ((parsing == 16) | (parsing == 18))
        
        # 背景：其他所有 (class 0, 16, 18)
        bg_mask = (~(np.isin(parsing, skin_indices + [16] + [17] + [18]))).astype(np.uint8)
        
        # 转换为 tensor
        def to_tensor(m):
            return torch.from_numpy(m).float().unsqueeze(0).unsqueeze(0)
        
        return {
            "skin": to_tensor(skin_mask),
            "hair": to_tensor(hair_mask),
            "clothes": to_tensor(clothes_mask),
            "bg": to_tensor(bg_mask)
        }





# ---------------------------------------------------------
# 组员扩展区域：
# class DeepLabSegmenter(BaseSegmenter):
#     def get_masks(...):
#         # 写你的 DeepLab 代码
#         pass
# ---------------------------------------------------------