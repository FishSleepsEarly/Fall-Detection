# extract_sk.py
import cv2
import mediapipe as mp
import numpy as np
import json

# 骨骼绘制函数
def draw_skeleton(frame, keypoints, threshold=0.5):
    """
    在帧上绘制骨架。

    参数:
    - frame: 当前帧的图像 (BGR格式)
    - keypoints: 关键点列表，每个关键点为[x, y, confidence]
    - threshold: 置信度阈值，低于该值的关键点将不被绘制
    """
    # 定义骨骼连接关系（基于 COCO）
    skeleton_pairs = [
        (0, 1), (0, 2),      # 鼻子到左眼，鼻子到右眼
        (1, 3), (2, 4),      # 左眼到左耳，右眼到右耳
        (0, 5), (0, 6),      # 鼻子到左肩，鼻子到右肩
        (5, 7), (7, 9),      # 左肩到左肘到左腕
        (6, 8), (8, 10),     # 右肩到右肘到右腕
        (5, 11), (6, 12),    # 左肩到左臀，右肩到右臀
        (11, 13), (13, 15),  # 左臀到左膝到左踝
        (12, 14), (14, 16)   # 右臀到右膝到右踝,
    ]

    # 定义骨骼颜色（BGR格式）
    color_map = {
        "head": (0, 255, 0),         # 绿色
        "left_arm": (255, 0, 0),     # 蓝色
        "right_arm": (255, 0, 255),  # 品红色
        "left_leg": (0, 0, 255),     # 红色
        "right_leg": (255, 255, 0)   # 青色
    }

    # 遍历每个人的关键点（MediaPipe Pose 仅支持单人）
    for person_id, person_keypoints in enumerate(keypoints):
        # 绘制关键点
        for x, y, conf in person_keypoints:
            if conf > threshold:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)  # 红色圆点

        # 绘制骨骼
        for (i, j) in skeleton_pairs:
            if person_keypoints[i][2] > threshold and person_keypoints[j][2] > threshold:
                pt1 = tuple(map(int, person_keypoints[i][:2]))
                pt2 = tuple(map(int, person_keypoints[j][:2]))

                # 根据连接关系选择颜色
                if (i, j) in [(0, 1), (1, 3), (0, 2), (2, 4)]:
                    color = color_map["head"]
                elif (i, j) in [(0, 5), (5, 7), (7, 9)]:
                    color = color_map["left_arm"]
                elif (i, j) in [(0, 6), (6, 8), (8, 10)]:
                    color = color_map["right_arm"]
                elif (i, j) in [(5, 11), (11, 13), (13, 15)]:
                    color = color_map["left_leg"]
                elif (i, j) in [(6, 12), (12, 14), (14, 16)]:
                    color = color_map["right_leg"]
                else:
                    color = (0, 255, 0)  # 默认绿色

                cv2.line(frame, pt1, pt2, color, 2)

    return frame

# 视频处理流程
def process_video(input_path, output_path, annotation_output_path):
    """
    使用 MediaPipe Pose 检测视频中的人体姿态，并在输出视频中绘制骨架，同时保存关键点。

    Args:
        input_path (str): 输入视频路径。
        output_path (str): 输出标注视频路径。
        annotation_output_path (str): 存储骨骼关键点的文件路径。
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {input_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    keypoints_data = {"frames": []}  # 用于存储关键点数据

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 转换为 RGB 格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 使用 MediaPipe 提取骨骼关键点
        results = pose.process(frame_rgb)
        frame_keypoints = []
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                frame_keypoints.append([lm.x, lm.y, lm.visibility])  # 保存 (x, y, 可见性)
            keypoints_data["frames"].append({"keypoints": frame_keypoints})

        # 在帧上绘制骨架（可选）
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        # 写入标注视频
        out.write(frame)

    cap.release()
    out.release()
    pose.close()

    # 保存关键点数据为 JSON 文件
    with open(annotation_output_path, 'w') as f:
        json.dump(keypoints_data, f)

    print(f"骨骼标注视频已保存至 {output_path}")
    print(f"关键点数据已保存至 {annotation_output_path}")

# 主程序入口
if __name__ == "__main__":
    # 直接在代码中指定输入和输出视频路径
     start_index = 11  # 起始序号
     end_index = 30    # 结束序号（包含）

    # 遍历序号范围，批量处理视频
     for idx in range(start_index, end_index + 1):
       input_video_path = f"raw_fall_videos/{idx}.mp4" 
       output_video_path = f"new_skvideos_fall/output_skeleton{idx}.mp4"  
       keypoints_path=f"json_dir/fall/keypoints_video{idx}.json"
       print(f"正在处理视频: {input_video_path} -> {output_video_path}")
    # 调用处理函数
       process_video(input_video_path, output_video_path, keypoints_path)