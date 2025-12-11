import mediapipe as mp
import cv2
import numpy as np
import torch
import os

def get_semantic_masks(image_path, width, height, method='mediapipe'):
    """
    获取语义分割 Mask
    
    Args:
        image_path: 图片路径
        width: 目标宽度
        height: 目标高度
        method: 分割方法 ('mediapipe' 或 'bisenet')
    
    Returns:
        dict: {"skin": tensor, "hair": tensor, "bg": tensor}
    """
    if method == 'bisenet':
        return _get_masks_bisenet(image_path, width, height)
    else:
        return _get_masks_mediapipe(image_path, width, height)


def _get_masks_bisenet(image_path, width, height):
    """
    BiSeNet 分割方法
    """
    from core.segmentation import BiSeNetSegmenter
    
    # 初始化 BiSeNet (使用单例模式避免重复加载)
    if not hasattr(_get_masks_bisenet, 'segmenter'):
        _get_masks_bisenet.segmenter = BiSeNetSegmenter(model_path='core/79999_iter.pth')
    
    masks = _get_masks_bisenet.segmenter.get_masks(image_path, width, height)
    
    # 保存调试图
    _save_debug_masks(image_path, masks)
    
    return masks


def _get_masks_mediapipe(image_path, width, height):
    """
    MediaPipe 增强版 (带调试与兜底)：
    1. 降低检测阈值，适应阿凡达/灰度图
    2. 如果检测不到脸，自动降级为 "Selfie 模式" (整个人=皮肤)
    3. 保存 Mask 预览图，方便排查
    """
    mp_face_mesh = mp.solutions.face_mesh
    mp_selfie = mp.solutions.selfie_segmentation
    
    # 1. 预处理
    if not os.path.exists(image_path):
        raise ValueError(f"文件不存在: {image_path}")
        
    image = cv2.imread(image_path)
    if image is None: 
        raise ValueError(f"无法读取图片 (可能是路径包含中文或损坏): {image_path}")
        
    image = cv2.resize(image, (width, height))
    # 灰度图增强：如果是单通道，强制转为 3 通道 BGR，否则 MediaPipe 会报错
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # --- Step A: 获取“整个人” (Person) ---
    # model_selection=1 (Landscape) 适合全身/半身
    with mp_selfie.SelfieSegmentation(model_selection=1) as selfie_seg:
        res = selfie_seg.process(image_rgb)
        if res.segmentation_mask is None:
            print(f"⚠️ 警告: SelfieSegmentation 在 {image_path} 上失败，使用全图 Mask")
            person_mask = np.ones((height, width), dtype=np.uint8)
        else:
            person_mask = (res.segmentation_mask > 0.1).astype(np.uint8) # 阈值降到 0.1 抢救边缘

    # --- Step B: 获取“纯脸部” & “下巴坐标” ---
    face_mask = np.zeros((height, width), dtype=np.uint8)
    chin_y = None # 初始化为 None 用于判断是否检测成功
    
    # 关键修改：confidence 降到 0.1，max_num_faces=1
    with mp_face_mesh.FaceMesh(
        static_image_mode=True, 
        max_num_faces=1, 
        refine_landmarks=True,
        min_detection_confidence=0.1, # <--- 极低阈值，强行检测阿凡达
        min_tracking_confidence=0.1
    ) as face_mesh:
        res = face_mesh.process(image_rgb)
        
        if res.multi_face_landmarks:
            landmarks = res.multi_face_landmarks[0].landmark
            
            # A. 绘制脸部 Mask
            points = []
            for lm in landmarks:
                points.append([int(lm.x * width), int(lm.y * height)])
            
            # 使用 ConvexHull 包裹所有点
            hull = cv2.convexHull(np.array(points, np.int32))
            cv2.fillConvexPoly(face_mask, hull, 1)
            
            # B. 找到下巴最低点 (索引 152)
            chin_y = int(landmarks[152].y * height)
        else:
            print(f"⚠️ 警告: FaceMesh 未检测到人脸: {image_path} (将启动兜底策略)")

    # --- Step C: 几何拆分与兜底 ---
    
    final_skin_mask = None
    final_hair_mask = None
    final_bg_mask = 1 - person_mask
    
    if chin_y is not None:
        # === 方案 1: 成功检测到脸，执行精细分割 ===
        remainder = person_mask - face_mask
        remainder = np.clip(remainder, 0, 1)
        
        y_coords = np.arange(height).reshape(height, 1).repeat(width, axis=1)
        split_line = chin_y + int(height * 0.05) # 下巴往下 5%
        
        # 身体 = 余数 && 在下巴下面
        body_skin_mask = np.logical_and(remainder == 1, y_coords > split_line).astype(np.uint8)
        # 头发 = 余数 && 在下巴上面
        hair_mask = np.logical_and(remainder == 1, y_coords <= split_line).astype(np.uint8)
        
        final_skin_mask = np.clip(face_mask + body_skin_mask, 0, 1)
        final_hair_mask = hair_mask
    else:
        # === 方案 2: 检测失败兜底 ===
        # 假设整个人都是皮肤 (宁可错杀，不可放过，防止 Loss=0)
        print(" -> 启动兜底: 将所有 Person 区域设为 Skin，Hair 设为 Empty")
        final_skin_mask = person_mask
        final_hair_mask = np.zeros((height, width), dtype=np.uint8)

    def to_tensor(m):
        return torch.from_numpy(m).float().unsqueeze(0).unsqueeze(0)

    masks = {
        "skin": to_tensor(final_skin_mask),
        "hair": to_tensor(final_hair_mask),
        "bg": to_tensor(final_bg_mask)
    }
    
    # 保存调试图
    _save_debug_masks(image_path, masks)
    
    return masks


def _save_debug_masks(image_path, masks):
    """
    保存调试用的 Mask 图片
    """
    debug_dir = "outputs/debug_masks"
    if not os.path.exists(debug_dir): 
        os.makedirs(debug_dir)
    
    base_name = os.path.basename(image_path)
    
    # 将 tensor 转为 numpy
    for region in ['skin', 'hair', 'bg']:
        mask_np = masks[region].squeeze().cpu().numpy()
        cv2.imwrite(f"{debug_dir}/debug_{base_name}_{region}.jpg", (mask_np * 255).astype(np.uint8))