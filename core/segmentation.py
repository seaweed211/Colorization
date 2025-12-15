import torch
import numpy as np
import cv2
import mediapipe as mp
import os
from PIL import Image

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

def prepare_image(image, input_size=(512, 512)):
    """
    Prepare an image for inference by resizing and normalizing it.
    完全复制自 face-parsing/inference.py
    
    Args:
        image: PIL Image to process
        input_size: Target size for resizing

    Returns:
        torch.Tensor: Preprocessed image tensor ready for model input
    """
    import torchvision.transforms as transforms
    
    # Resize the image
    resized_image = image.resize(input_size, resample=Image.BILINEAR)

    # Define transformation pipeline
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    # Apply transformations
    image_tensor = transform(resized_image)
    image_batch = image_tensor.unsqueeze(0)

    return image_batch


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
        self.debug_mode = False  # 调试模式：保存原始 19 类 parsing

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
        完全按照 face-parsing/inference.py 的流程实现（逐行复制）
        
        Args:
            image_path: 输入图片路径
            width: 目标宽度
            height: 目标高度
            
        Returns:
            dict: 包含 skin, hair, clothes, bg 四个 mask 的字典
        """
        from PIL import Image
        
        # ========== 以下代码完全复制自 face-parsing/inference.py ==========
        
        # Load and process the image
        image = Image.open(image_path).convert('RGB')

        # Store original image resolution
        # 【修复】使用传入的 width, height 参数，而不是图片原始尺寸
        # 这样才能和 content_img 的尺寸匹配
        original_size = (width, height)  # (width, height)

        # Prepare image for inference
        image_batch = prepare_image(image).to(self.device)

        # Run inference (注意：face-parsing 使用 @torch.no_grad() 装饰器)
        with torch.no_grad():
            output = self.net(image_batch)[0]  # feat_out, feat_out16, feat_out32 -> use feat_out for inference only
            predicted_mask = output.squeeze(0).cpu().numpy().argmax(0)

        # Convert mask to PIL Image for resizing
        mask_pil = Image.fromarray(predicted_mask.astype(np.uint8))

        # Resize mask back to original image resolution
        restored_mask = mask_pil.resize(original_size, resample=Image.NEAREST)

        # Convert back to numpy array
        parsing = np.array(restored_mask)
        
        # ========== face-parsing 代码结束 ==========
        
        # 【调试】保存原始 19 类的 parsing 结果
        if hasattr(self, 'debug_mode') and self.debug_mode:
            self._save_raw_parsing(parsing, image, width, height)

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
    
    def _save_raw_parsing(self, parsing, image, width, height):
        """
        保存原始 19 类 parsing 结果用于调试
        
        Args:
            parsing: numpy array of parsing result (shape: H x W)
            image: PIL Image object (原始图像)
            width: 目标宽度 (传入的参数)
            height: 目标高度 (传入的参数)
        """
        import os
        
        # 创建调试目录
        debug_dir = 'outputs/debug_parsing'
        os.makedirs(debug_dir, exist_ok=True)
        
        # 获取 parsing 的实际尺寸
        parsing_height, parsing_width = parsing.shape
        
        # Resize image to match parsing size
        # 注意：PIL.resize 参数是 (width, height)，parsing.shape 是 (height, width)
        image_resized = image.resize((parsing_width, parsing_height), resample=Image.BILINEAR)
        
        # 可视化 parsing（使用与 face-parsing 相同的颜色映射）
        from utils.common import vis_parsing_maps
        
        # 使用时间戳作为文件名
        import time
        timestamp = int(time.time())
        save_path = os.path.join(debug_dir, f'parsing19_{timestamp}.jpg')
        
        vis_parsing_maps(image_resized, parsing, save_image=True, save_path=save_path)
        print(f"    -> 原始 19 类 parsing 已保存: {save_path}")
        
        # 统计每个类别的像素数
        unique, counts = np.unique(parsing, return_counts=True)
        class_names = ['bg', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye',
                      'eye_g', 'l_ear', 'r_ear', 'ear_r', 'nose', 'mouth',
                      'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
        
        print(f"    -> 类别统计:")
        for cls_id, count in zip(unique, counts):
            if cls_id < len(class_names):
                print(f"       {cls_id:2d} ({class_names[cls_id]:8s}): {count:6d} pixels")


# ---------------------------------------------------------
# 组员扩展区域：
# class DeepLabSegmenter(BaseSegmenter):
#     def get_masks(...):
#         # 写你的 DeepLab 代码
#         pass
# ---------------------------------------------------------