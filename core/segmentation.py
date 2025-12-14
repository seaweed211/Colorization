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


# class BiSeNetSegmenter(BaseSegmenter):
#     """
#     BiSeNet 人脸解析分割器：使用 BiSeNet 进行 19 类人脸解析，
#     并将结果合并为 hair/skin/bg 三类 mask。
#     """
#     def __init__(self, model_path='core/79999_iter.pth'):
#         """
#         初始化 BiSeNet 模型
#         Args:
#             model_path: 模型权重文件路径
#         """
#         from .model import BiSeNet
#         from collections import OrderedDict
#
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.net = BiSeNet(n_classes=19)
#         self.net.to(self.device)
#
#         # 加载模型权重
#         if os.path.exists(model_path):
#             state_dict = torch.load(model_path, map_location=self.device)
#             # 处理 DataParallel 的 module 前缀
#             new_state_dict = OrderedDict()
#             for k, v in state_dict.items():
#                 name = k[7:] if k.startswith('module.') else k
#                 new_state_dict[name] = v
#             self.net.load_state_dict(new_state_dict)
#             self.net.eval()
#             print(f"BiSeNet 模型加载成功: {model_path}")
#         else:
#             raise FileNotFoundError(f"模型文件不存在: {model_path}")
#
#     def get_masks(self, image_path, width, height):
#         """
#         使用 BiSeNet 获取 hair/skin/bg 三类 mask
#
#         BiSeNet 19 类定义:
#         0:背景, 1:脸, 2-9:五官, 10:鼻, 11-13:嘴, 14-15:脖子, 16:衣, 17:发, 18:帽
#
#         合并规则:
#         - hair: class 17 (头发)
#         - skin: class 1-15 (脸部、五官、脖子)
#         - bg: class 0, 16, 18 (背景、衣服、帽子)
#         """
#         # 读取图片
#         img = cv2.imread(image_path)
#         if img is None:
#             raise ValueError(f"无法读取图片: {image_path}")
#
#         # 转换为 RGB
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#         # BiSeNet 需要 512x512 输入
#         img_512 = cv2.resize(img_rgb, (512, 512))
#
#         # 归一化
#         img_tensor = torch.from_numpy(img_512).float().permute(2, 0, 1) / 255.0
#         mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
#         std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
#         img_tensor = (img_tensor - mean) / std
#         img_tensor = img_tensor.unsqueeze(0).to(self.device)
#
#         # 推理
#         with torch.no_grad():
#             out = self.net(img_tensor)[0]
#             parsing = out.squeeze(0).cpu().numpy().argmax(0)
#
#         # 还原到目标尺寸
#         parsing = cv2.resize(parsing.astype(np.uint8), (width, height),
#                            interpolation=cv2.INTER_NEAREST)
#
#         # 创建4类 mask
#         # 头发
#         hair_mask = (parsing == 17).astype(np.uint8)
#
#         # 皮肤：脸部 + 五官 + 脖子 (class 1-15)
#         skin_indices = list(range(1, 16))
#         skin_mask = np.isin(parsing, skin_indices).astype(np.uint8)
#
#         # 衣服
#         clothes_mask = ((parsing == 16) | (parsing == 18))
#
#         # 背景：其他所有 (class 0, 16, 18)
#         bg_mask = (~(np.isin(parsing, skin_indices + [16] + [17] + [18]))).astype(np.uint8)
#
#         # 转换为 tensor
#         def to_tensor(m):
#             return torch.from_numpy(m).float().unsqueeze(0).unsqueeze(0)
#
#         return {
#             "skin": to_tensor(skin_mask),
#             "hair": to_tensor(hair_mask),
#             "clothes": to_tensor(clothes_mask),
#             "bg": to_tensor(bg_mask)
#         }

class BiSeNetSegmenter(BaseSegmenter):
    """
    BiSeNet 人脸解析分割器 (基于 yakhyo/face-parsing 实现)
    使用 ResNet34 作为骨干网络进行 19 类人脸解析，
    并将结果合并为 skin/hair/clothes/bg 四类 mask。
    """

    def __init__(self, model_path='core/weights/resnet34.pt', backbone='resnet34'):
        """
        初始化 BiSeNet 模型
        Args:
            model_path: 模型权重文件路径 (例如 resnet34.pt)
            backbone: 骨干网络类型 ('resnet18' 或 'resnet34')
        """
        # 使用 core/bisenet.py 中的 BiSeNet 实现
        try:
            from .bisenet import BiSeNet
        except ImportError:
            raise ImportError(
                "未找到 bisenet.py。请确保 core/bisenet.py 存在。")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = (512, 512)  # BiSeNet 标准输入尺寸

        # 初始化模型 (注意：bisenet.py 中参数名是 num_classes 和 backbone_name)
        self.net = BiSeNet(num_classes=19, backbone_name=backbone)
        self.net.to(self.device)

        # 加载模型权重
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.net.load_state_dict(state_dict)
            self.net.eval()
            print(f"BiSeNet ({backbone}) 模型加载成功: {model_path}")
        except Exception as e:
            raise RuntimeError(f"加载权重失败，请检查权重文件是否与 backbone={backbone} 匹配。错误: {e}")

    def get_masks(self, image_path, width, height):
        """
        使用 BiSeNet 获取 mask 并按照 CelebAMask-HQ 标准合并
        
        Args:
            image_path: 输入图片路径
            width: 目标宽度
            height: 目标高度
            
        Returns:
            dict: 包含 skin, hair, clothes, bg 四个 mask 的字典
        """
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")

        # 转换为 RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 预处理：Resize 到 512x512 (模型训练时的固定尺寸)
        img_512 = cv2.resize(img_rgb, self.input_size, interpolation=cv2.INTER_LINEAR)

        # 归一化 (使用 ImageNet 标准均值和方差)
        img_tensor = torch.from_numpy(img_512).float().permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.to(self.device)  # 先移到 GPU
        
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        img_tensor = img_tensor.unsqueeze(0)  # 添加 batch 维度

        # 推理
        with torch.no_grad():
            # BiSeNet 返回三个输出: feat_out, feat_out16, feat_out32
            # 我们只使用第一个 (最高分辨率的输出)
            out = self.net(img_tensor)

            # 如果 out 是 tuple，取第一个元素
            if isinstance(out, (list, tuple)):
                out = out[0]

            # Argmax 获取分类索引 [512, 512]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

        # 还原到目标尺寸 (使用最近邻插值保持类别整数)
        # 注意：cv2.resize 的参数顺序是 (width, height)
        parsing = cv2.resize(
            parsing.astype(np.uint8), 
            (width, height),
            interpolation=cv2.INTER_NEAREST
        )

        # ==========================================================
        # 类别合并逻辑 (CelebAMask-HQ 19类标准)
        # 0:bg, 1:skin, 2:l_brow, 3:r_brow, 4:l_eye, 5:r_eye,
        # 6:eye_g, 7:l_ear, 8:r_ear, 9:ear_r, 10:nose, 11:mouth,
        # 12:u_lip, 13:l_lip, 14:neck, 15:neck_l, 16:cloth,
        # 17:hair, 18:hat
        # ==========================================================

        # 1. 头发 (Hair): 17
        hair_mask = (parsing == 17).astype(np.uint8)

        # 2. 皮肤 (Skin): 1(脸) + 2-6(眉毛眼睛眼镜) + 7-9(耳朵耳环) + 10-13(鼻子嘴唇) + 14-15(脖子)
        # 注意：耳朵 (7:左耳, 8:右耳, 9:耳环) 已包含在皮肤类别中
        skin_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        skin_mask = np.isin(parsing, skin_indices).astype(np.uint8)

        # 3. 衣服 (Clothes): 16(衣服) + 18(帽子，通常归类为装饰或衣服)
        clothes_mask = np.isin(parsing, [16, 18]).astype(np.uint8)

        # 4. 背景 (Bg): 排除以上所有的区域
        # 逻辑：不是头发、不是皮肤、不是衣服，就是背景
        bg_mask = (~(hair_mask.astype(bool) | skin_mask.astype(bool) | clothes_mask.astype(bool))).astype(np.uint8)

        # 转换为 Tensor 格式 [1, 1, H, W]
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