import argparse
import torch
import cv2
import numpy as np
import os
from core.segmentation import MediaPipeSegmenter, BiSeNetSegmenter,GlobalSegmenter
from core.initialization import AdaINRegionAware, GrayStart
from core.solvers import OptimizationSolver
from core.guidance import CLIPGuidance
from utils.image_utils import load_image_with_ratio, transfer_color_presolve_luminance, save_result
import clip # 只需要tokenize用

# 配置参数
PROMPT_POOLS = {
    "skin": ["blue skin", "Avatar Na'vi skin", "dark blue skin"],
    "hair": ["black hair", "braided hair"],
    "bg": ["forest", "jungle", "dark background"]
}

def _save_mask_visualization(image_path, masks, width, height, prefix=''):
    """
    保存 mask 可视化图，用于调试分割效果
    """
    # 读取原图
    img = cv2.imread(image_path)
    if img is None:
        return
    img = cv2.resize(img, (width, height))
    
    # 提取 mask (从 tensor 转为 numpy)
    skin = masks['skin'].squeeze().cpu().numpy()
    hair = masks['hair'].squeeze().cpu().numpy()
    bg = masks['bg'].squeeze().cpu().numpy()
    
    # 创建彩色 mask 叠加图
    overlay = img.copy()
    colored_mask = np.zeros_like(img)
    
    # 绿色 = 皮肤, 红色 = 头发, 蓝色 = 背景
    colored_mask[skin > 0.5] = [0, 255, 0]  # 绿色
    colored_mask[hair > 0.5] = [0, 0, 255]  # 红色
    colored_mask[bg > 0.5] = [255, 128, 0]  # 蓝色
    
    # 半透明叠加
    result = cv2.addWeighted(img, 0.6, colored_mask, 0.4, 0)
    
    # 保存
    output_dir = 'outputs/debug_masks'
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.basename(image_path).split('.')[0]
    output_path = os.path.join(output_dir, f'{prefix}_{base_name}_mask_vis.jpg')
    cv2.imwrite(output_path, result)
    
    # 也保存单独的 mask
    cv2.imwrite(os.path.join(output_dir, f'{prefix}_{base_name}_skin.jpg'), (skin * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, f'{prefix}_{base_name}_hair.jpg'), (hair * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, f'{prefix}_{base_name}_bg.jpg'), (bg * 255).astype(np.uint8))
    
    print(f"    -> Mask 可视化已保存: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', default='inputs/in1.jpg')
    parser.add_argument('--style', default='styles/ref1.jpg')
    parser.add_argument('--output', default='outputs/out1.jpg')
    parser.add_argument('--seg_method', default='bisenet', help='分割方法: mediapipe | bisenet | global | schp')
    parser.add_argument('--init_method', default='adain', help='初始化: adain | gray')
    parser.add_argument('--use_clip', action='store_true', help='是否使用 CLIP')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 模块工厂 (Factory Pattern)
    # 你的组员可以在这里注册他们的新模型
 
    segmenters = {
    'mediapipe': MediaPipeSegmenter(),
    'bisenet': BiSeNetSegmenter(model_path='core/79999_iter.pth'),
    'global': GlobalSegmenter()
}
    initializers = {
        'adain': AdaINRegionAware(),
        'gray': GrayStart()
    }
    
    # 实例化当前选择的模块
    seg_module = segmenters[args.seg_method]
    init_module = initializers[args.init_method]
    solver_module = OptimizationSolver()

    # 2. 加载数据
    print("--- 1. 加载图片 ---")
    content_img, cw, ch = load_image_with_ratio(args.content)
    
    # 先加载参考图 (尺寸可能不一致)
    style_img_raw, _, _ = load_image_with_ratio(args.style, size=max(cw, ch))
    
    # 【修复点】强制将参考图缩放到与原图完全一致 (ch, cw)
    # 使用 interpolate 进行调整，确保后续 Mask 和 Image 像素一一对应
    style_img = torch.nn.functional.interpolate(style_img_raw, size=(ch, cw), mode='bicubic')

    # 3. 分割
    print(f"--- 2. 执行分割 ({args.seg_method}) ---")
    c_masks = seg_module.get_masks(args.content, cw, ch)
    s_masks = seg_module.get_masks(args.style, cw, ch) # 假设参考图也用同样方法分割
    
    # 保存 mask 可视化（用于调试）
    _save_mask_visualization(args.content, c_masks, cw, ch, prefix='content')
    _save_mask_visualization(args.style, s_masks, cw, ch, prefix='style')
    
    # 把 mask 转到 GPU
    for d in [c_masks, s_masks]:
        for k in d: d[k] = d[k].to(device)

    # 4. 初始化
    print(f"--- 3. 执行初始化 ({args.init_method}) ---")
    init_img = init_module.process(content_img, style_img, c_masks, s_masks)

    # 5. 准备 Prompt 
    active_prompts = {}  # 先初始化为空字典，防止后面报错
    guidance_module = None # 默认为 None

    if args.use_clip:
        print("--- 3.5 启用 CLIP 引导 ---")
        guidance_module = CLIPGuidance()
        
        print("--- 4. 准备 CLIP 语义 ---")
        # 只有当 guidance_module 存在时，才去调用它的方法
        active_prompts = guidance_module.get_best_prompts(style_img, s_masks, PROMPT_POOLS)
        print(f"    -> 使用的提示词: {active_prompts}")
        
    else:
        print("--- 3.5 禁用 CLIP (纯风格迁移) ---")
        # 这里什么都不用做，guidance_module 是 None，active_prompts 是空字典 {}



    # 6. 运行求解器
    print("--- 5. 开始风格迁移优化 ---")
    final_raw = solver_module.run(init_img, style_img, c_masks, s_masks, guidance_module, active_prompts)

    # 7. 后处理
    print("--- 6. 后处理与保存 ---")
    final_result = transfer_color_presolve_luminance(content_img, final_raw)
    save_result(final_result, args.output)
    print("Done!")

if __name__ == '__main__':
    main()