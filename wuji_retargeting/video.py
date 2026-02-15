import cv2
import os

def images_to_video(image_folder, output_video_path, frame_rate=30):
    # 获取所有 PNG 图片文件
    images = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    images.sort()  # 按文件名排序，以确保按顺序加载图片

    if len(images) == 0:
        print("文件夹中没有 PNG 图片！")
        return

    # 读取第一张图片获取视频的宽高
    first_image_path = os.path.join(image_folder, images[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape

    # 设置视频输出格式
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用 XVID 编码
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    # 将每张图片按顺序写入视频
    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)
        video_writer.write(image)

    # 释放视频写入器
    video_writer.release()
    print(f"视频已经保存到: {output_video_path}")

if __name__ == "__main__":
    # 设置图片文件夹路径和输出视频路径
    image_folder = "/home/heng/heng/example2_output/overlays"  # 替换成你的图片文件夹路径
    output_video_path = "output_video.mp4"  # 输出视频文件名

    images_to_video(image_folder, output_video_path)
