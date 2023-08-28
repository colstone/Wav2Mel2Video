import os
import numpy as np
import librosa
import librosa.display
import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt

# 获取用户输入的音频文件路径
audio_file_path = input("请输入音频文件的路径（包括文件名和后缀）：")

# 检查文件是否存在
if not os.path.exists(audio_file_path):
    print("文件路径不存在，请检查路径是否正确。")
    exit()

# 获取音频文件所在的目录和文件名
audio_dir, audio_filename = os.path.split(audio_file_path)
audio_name, audio_extension = os.path.splitext(audio_filename)

# 创建同名的 temp 文件夹
temp_folder_path = os.path.join(audio_dir, audio_name + "_temp")
if not os.path.exists(temp_folder_path):
    os.makedirs(temp_folder_path)
    print(f"已创建文件夹 {temp_folder_path} 用于存放生成的 Mel 图。")
else:
    print(f"文件夹 {temp_folder_path} 已存在。")

# 加载音频文件
y, sr = librosa.load(audio_file_path)

# 设置帧率和高斯噪声的步数
frame_rate = 25

# 设置自定义的生成图片张数
custom_num_steps = 3000

# 设置不同加噪的范围和参数
linear_start = 0
linear_end = 750
log_start = 751
log_end = 2500
gaussian_start = 2501
gaussian_end = custom_num_steps

# 用 tqdm 添加进度条来显示生成 Mel 图的进度
for step in tqdm(range(custom_num_steps + 1), desc="Generating Mel Spectrograms"):
    # 线性加噪
    if linear_start <= step <= linear_end:
        linear_factor = step / linear_end
        noisy_y = y + np.random.normal(0, 0.01 * linear_factor, len(y))
    # 渐变为对数加噪
    elif log_start <= step <= log_end:
        log_factor = 30 * np.log(1 + (step - log_start)) / np.log(1 + (log_end - log_start))
        noisy_y = y + np.random.normal(0, 0.01 * log_factor, len(y))
    # 渐变为高斯噪声
    else:
        gaussian_factor = 1 + (step - gaussian_start) / (gaussian_end - gaussian_start)
        noisy_y = y + np.random.normal(0, 0.01 * gaussian_factor, len(y))
    
    # 计算 Mel 频谱
    mel_db = librosa.feature.melspectrogram(y=noisy_y, sr=sr)
    mel_db = librosa.power_to_db(mel_db, ref=np.max)

    # 生成 Mel 图
    plt.figure(figsize=(10, 6), dpi=200)
    librosa.display.specshow(mel_db, sr=sr, x_axis=None, y_axis=None, cmap=plt.cm.viridis)
    plt.axis('off')
    mel_filename = f"mel_{audio_name}_{step:03d}.png"
    mel_filepath = os.path.join(temp_folder_path, mel_filename)
    plt.savefig(mel_filepath, bbox_inches='tight', pad_inches=0)
    plt.close()

# 保存 Mel 图并生成视频
images = []
for step in range(custom_num_steps + 1):
    mel_filename = f"mel_{audio_name}_{step:03d}.png"
    mel_filepath = os.path.join(temp_folder_path, mel_filename)
    
    try:
        image = imageio.imread(mel_filepath)
        images.append(image)
    except FileNotFoundError:
        print(f"跳过缺失的图片: {mel_filepath}")
        continue

# 创建视频写入器
output_video_path = os.path.join(audio_dir, f"{audio_name}_output.mp4")
fourcc = "mp4v"  # 编码格式
frame_rate = 25

# 保存正向扩散视频
with imageio.get_writer(output_video_path, fps=frame_rate) as video_writer:
    for image in images:
        video_writer.append_data(image)

# 反向扩散视频
reversed_images = list(reversed(images))
with imageio.get_writer(output_video_path, fps=frame_rate) as video_writer:
    for image in reversed_images:
        video_writer.append_data(image)

print(f"已生成视频 {output_video_path}")