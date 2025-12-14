import os
import subprocess
import argparse
from tqdm import tqdm  # 如果没有安装 tqdm，可以运行 pip install tqdm，或者删掉相关代码

# --- 配置区域 ---
INPUT_DIR = 'in'    # 输入图片文件夹
STYLE_DIR = 'style'    # 参考图片文件夹
OUTPUT_DIR = 'out'  # 结果保存文件夹

# 这里设置你想传给 main.py 的固定参数
# 比如是否开启 CLIP，使用什么分割方法等
DEFAULT_ARGS = [
    '--use_clip',              # 开启 CLIP
    '--seg_method', 'bisenet', # 分割方法
    '--init_method', 'adain'   # 初始化方法
]

def get_image_files(directory):
    """获取文件夹下所有的图片文件，并按文件名排序"""
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    files = [f for f in os.listdir(directory) if f.lower().endswith(valid_extensions)]
    # 排序非常重要，保证 in1 对应 ref1, in2 对应 ref2
    files.sort()
    return files

def main():
    # 1. 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. 获取文件列表
    content_files = get_image_files(INPUT_DIR)
    style_files = get_image_files(STYLE_DIR)

    # 3. 检查数量
    count = min(len(content_files), len(style_files))
    if count == 0:
        print("错误：inputs 或 styles 文件夹为空，或者没有匹配的图片文件！")
        return

    print(f"找到 {len(content_files)} 张内容图, {len(style_files)} 张风格图。")
    print(f"即将处理前 {count} 对图片...\n")

    # 4. 循环处理
    # 使用 tqdm 显示进度条 (如果你不想用 tqdm，可以将 range(count) 改为 enumerate)
    for i in range(count):
        c_name = content_files[i]
        s_name = style_files[i]
        
        # 构造完整路径
        c_path = os.path.join(INPUT_DIR, c_name)
        s_path = os.path.join(STYLE_DIR, s_name)
        
        # 构造输出文件名：out_原文件名.jpg
        # 或者你可以改成 out1.jpg, out2.jpg
        out_name = f"out_{os.path.splitext(c_name)[0]}.jpg"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        print(f"正在处理 [{i+1}/{count}]: Content={c_name} | Style={s_name}")

        # 5. 构造命令行命令
        # 相当于在终端输入 python main.py --content ...
        cmd = [
            'python', 'main.py',
            '--content', c_path,
            '--style', s_path,
            '--output', out_path
        ] + DEFAULT_ARGS

        # 6. 调用系统命令执行
        try:
            # subprocess.run 会等待 main.py 跑完再跑下一张
            subprocess.run(cmd, check=True)
            print(f"--> 保存至: {out_path}\n")
        except subprocess.CalledProcessError as e:
            print(f"XXX 处理出错: {c_name} (Error Code: {e.returncode})")
            continue
        except KeyboardInterrupt:
            print("\n用户手动停止。")
            break

if __name__ == '__main__':
    main()