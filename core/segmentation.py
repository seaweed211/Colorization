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


# ---------------------------------------------------------
# 组员扩展区域：
# class DeepLabSegmenter(BaseSegmenter):
#     def get_masks(...):
#         # 写你的 DeepLab 代码
#         pass
# ---------------------------------------------------------