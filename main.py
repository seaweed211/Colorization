import argparse
import torch
import cv2
import numpy as np
import os
from core.segmentation import MediaPipeSegmenter, BiSeNetSegmenter, GlobalSegmenter
from core.initialization import AdaINRegionAware, GrayStart
from core.solvers import OptimizationSolver
from core.guidance import CLIPGuidance
from utils.image_utils import load_image_with_ratio, transfer_color_presolve_luminance, save_result, smart_color_merge, refine_masks
import clip # 只需要tokenize用


def _save_mask_visualization(image_path, masks, width, height, prefix=''):
    """
    保存 mask 可视化图，用于调试分割效果
    """
    # 读取原图
    img = cv2.imread(image_path)
    if img is None:
        return
    
    # 提取 mask (从 tensor 转为 numpy)
    skin = masks['skin'].squeeze().cpu().numpy()
    hair = masks['hair'].squeeze().cpu().numpy()
    clothes = masks.get('clothes', masks['bg']).squeeze().cpu().numpy()  # 如果没有clothes，使用bg
    bg = masks['bg'].squeeze().cpu().numpy()
    
    # 获取 mask 的实际尺寸 (height, width)
    mask_height, mask_width = skin.shape
    
    # Resize 图片以匹配 mask 尺寸
    # 注意：cv2.resize 参数是 (width, height)
    img = cv2.resize(img, (mask_width, mask_height))
    
    # 创建彩色 mask 叠加图
    overlay = img.copy()
    colored_mask = np.zeros_like(img)
    
    # 绿色 = 皮肤, 红色 = 头发, 黄色 = 衣服, 蓝色 = 背景
    colored_mask[skin > 0.5] = [0, 255, 0]  # 绿色
    colored_mask[hair > 0.5] = [0, 0, 255]  # 红色
    colored_mask[clothes > 0.5] = [0, 255, 255]  # 黄色
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
    cv2.imwrite(os.path.join(output_dir, f'{prefix}_{base_name}_clothes.jpg'), (clothes * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, f'{prefix}_{base_name}_bg.jpg'), (bg * 255).astype(np.uint8))
    
    print(f"    -> Mask 可视化已保存: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', default='inputs/in2.jpg')
    parser.add_argument('--style', default='styles/ref2.jpg')
    parser.add_argument('--output', default='outputs/out12.jpg')
    parser.add_argument('--seg_method', default='bisenet', help='分割方法: mediapipe | bisenet | global | schp')
    parser.add_argument('--init_method', default='adain', help='初始化: adain | gray')
    parser.add_argument('--use_clip', action='store_true', help='是否使用 CLIP')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 模块工厂 (Factory Pattern)
    # 你的组员可以在这里注册他们的新模型
 
    segmenters = {
    'mediapipe': MediaPipeSegmenter(),
    'bisenet':  BiSeNetSegmenter(model_path='core/weights/resnet34.pt', backbone='resnet34'),
    'global': GlobalSegmenter()
}
    
    # 启用 BiSeNet 调试模式（保存原始 19 类 parsing）
    if args.seg_method == 'bisenet':
        segmenters['bisenet'].debug_mode = True
    initializers = {
        'adain': AdaINRegionAware(),
        'gray': GrayStart()
    }
    
    # 实例化当前选择的模块
    seg_module = segmenters[args.seg_method]
    init_module = initializers[args.init_method]
    solver_module = OptimizationSolver(

    )

    # 2. 加载数据
    print("--- 1. 加载图片 ---")
    content_img, cw, ch = load_image_with_ratio(args.content)
    
    # 先加载参考图 (尺寸可能不一致)
    style_img_raw, _, _ = load_image_with_ratio(args.style, size=max(cw, ch))
    
    # 【修复点】强制将参考图缩放到与原图完全一致 (ch, cw)
    # 使用 interpolate 进行调整，确保后续 Mask 和 Image 像素一一对应
    style_img = torch.nn.functional.interpolate(style_img_raw, size=(ch, cw), mode='bicubic')

    # 3. 分割
    # 【关键修复】get_masks 的参数是 (width, height)，但 tensor 的 size 是 (height, width)
    # content_img.shape = [1, 3, ch, cw]，所以 mask 也应该是 (ch, cw)
    # 但 get_masks(path, width, height) 内部会创建 (height, width) 的 mask
    # 所以这里传参应该是 get_masks(path, cw, ch) 来得到 (ch, cw) 的 mask
    print(f"--- 2. 执行分割 ({args.seg_method}) ---")
    c_masks = seg_module.get_masks(args.content, cw, ch)
    s_masks = seg_module.get_masks(args.style, cw, ch) # 假设参考图也用同样方法分割

    # print("--- 2.5 优化 Mask (消除锯齿与缝隙) ---")
    # # dilate_iter=5 能够很好地填补脸和头发的缝隙
    # # blur_kernel=21 能让边缘非常平滑
    # c_masks = refine_masks(c_masks_raw, dilate_iter=5, blur_kernel=21)
    # s_masks = refine_masks(s_masks_raw, dilate_iter=5, blur_kernel=21)
    
    # # 保存 mask 可视化（用于调试）
    # # 注意：现在保存的是优化后的 mask，你应该能看到边缘变虚了
    # _save_mask_visualization(args.content, c_masks, cw, ch, prefix='content_refined')
    
    # # 把 mask 转到 GPU
    # for d in [c_masks, s_masks]:
    #     for k in d: d[k] = d[k].to(device)

    
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
    guidance_module = None # 默认为 None
    target_features = None # 默认为 None

    if args.use_clip:
        print("--- 3.5 启用 CLIP 引导 ---")
        guidance_module = CLIPGuidance(device)
        
        print("--- 4. 提取参考图 CLIP 特征 ---")
        # 直接传入 style_img 和 s_masks，提取参考图的视觉特征
        target_features = guidance_module.get_image_embeddings(style_img, s_masks)
        
        print(f"    -> 已提取特征区域: {list(target_features.keys())}")
        
    else:
        print("--- 3.5 禁用 CLIP (纯风格迁移) ---")



    # 6. 运行求解器
    print("--- 5. 开始风格迁移优化 ---")
    final_raw = solver_module.run(init_img, style_img, c_masks, s_masks, guidance_module, target_features)
    final_result_numpy = smart_color_merge(content_img, final_raw)

    # orig_img_np = content_img.squeeze(0).cpu().permute(1, 2, 0).numpy()
    # orig_img_np = np.clip(orig_img_np * 255, 0, 255).astype(np.uint8)
    # orig_img_np = cv2.cvtColor(orig_img_np, cv2.COLOR_RGB2BGR)

    # # 3. 准备“透明”掩模 (Alpha Mask)
    # # 我们把需要上色的区域（比如皮肤+头发+衣服）合并且羽化
    # # 只要是有 Mask 的地方就是前景，其他地方保持原图
    
    # # 提取 mask tensor
    # skin_mask = c_masks['skin'] 
    # hair_mask = c_masks['hair'] if 'hair' in c_masks else torch.zeros_like(skin_mask)
    # # 如果有衣服也可以加: clothes_mask = ...
    
    # # 合并 Mask (逻辑或)
    # combined_mask = torch.max(skin_mask, hair_mask) 
    
    # # Tensor -> Numpy (0.0 - 1.0)
    # alpha = combined_mask.squeeze().cpu().numpy().astype(np.float32)
    
    # # 【关键步骤】制作“透明渐变”
    # # 如果之前 refine_masks 里的 blur_kernel 不够大，这里可以再模糊一次 mask
    # # 使得边缘产生 0.8, 0.5, 0.2 这种半透明过渡
    # alpha = cv2.GaussianBlur(alpha, (21, 21), 0) 
    
    # # 扩展维度以匹配图片通道 (H, W) -> (H, W, 3)
    # alpha = np.expand_dims(alpha, axis=2)

    # # 4. 执行 Alpha 融合 (Blender)
    # # 公式：Result = Colored * Alpha + Original * (1 - Alpha)
    
    # # 将图片转为 float32 进行计算
    # fg = full_colored_img.astype(np.float32)
    # bg = orig_img_np.astype(np.float32)
    
    # # 融合
    # final_blended = fg * alpha + bg * (1.0 - alpha)
    
    # # 转回 uint8
    # final_blended = np.clip(final_blended, 0, 255).astype(np.uint8)

    # # 5. 保存
    # cv2.imwrite(args.output, final_blended)




    # 7. 后处理
    print("--- 6. 后处理与保存 ---")
    
    cv2.imwrite(args.output, final_result_numpy)    #!!
    
    # final_result = transfer_color_presolve_luminance(content_img, final_raw)
    # save_result(final_result, args.output)
    print("Done!")

if __name__ == '__main__':
    main()