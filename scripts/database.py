import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2
import math
import json
import numpy as np


class FallDetectionVideoDataset(torch.utils.data.Dataset):
    def __init__(self, video_dir, json_dir):
        self.video_dir = video_dir
        self.json_dir = os.path.abspath(json_dir)
        self.video_paths = []
        self.labels = []

        for label, category in enumerate(["notfall", "fall"]):
            category_video_dir = os.path.join(video_dir, category)
            for video_file in os.listdir(category_video_dir):
                if video_file.endswith(".mp4"):
                    self.video_paths.append(os.path.join(category_video_dir, video_file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        video_name = os.path.basename(video_path)
        json_name = os.path.splitext(video_name)[0] + ".json"
        category = "fall" if label == 1 else "notfall"
        json_path = os.path.join(self.json_dir, category, json_name)

        keypoints = self.extract_keypoints_from_annotations(json_path)

        if keypoints is None or len(keypoints) == 0:
            print(f"Warning: Keypoints missing or empty for JSON file: {json_path}")
            keypoints = [[0.0] * 33]

        # 插值处理关键点
        keypoints = self.interpolate_missing_keypoints(keypoints)

        angles = self.calculate_angles(keypoints)

        if angles is None or len(angles) == 0:
            print(f"Warning: Angles missing or empty for keypoints from JSON: {json_path}")
            angles = [0.0] * 3

        return torch.tensor(angles, dtype=torch.float32), label

    def extract_keypoints_from_annotations(self, annotation_path):
        with open(annotation_path, 'r') as f:
            keypoints_data = json.load(f)

        keypoints = []
        for frame_data in keypoints_data["frames"]:
            keypoints.append(frame_data["keypoints"])

        return keypoints

    def calculate_angles(self, keypoints):
        def compute_angle(p1, p2, p3):
            a = (p1[0] - p2[0], p1[1] - p2[1])
            b = (p3[0] - p2[0], p3[1] - p2[1])

            dot_product = a[0] * b[0] + a[1] * b[1]
            magnitude_a = math.sqrt(a[0]**2 + a[1]**2)
            magnitude_b = math.sqrt(b[0]**2 + b[1]**2)

            if magnitude_a * magnitude_b == 0:
                return 0.0

            cos_angle = dot_product / (magnitude_a * magnitude_b)
            angle = math.degrees(math.acos(min(1, max(-1, cos_angle))))
            return angle

        angles = []
        for frame_keypoints in keypoints:
            if len(frame_keypoints) < 33:
                continue

            left_hip = frame_keypoints[11][:2]
            left_knee = frame_keypoints[13][:2]
            left_ankle = frame_keypoints[15][:2]
            right_hip = frame_keypoints[12][:2]
            right_knee = frame_keypoints[14][:2]
            right_ankle = frame_keypoints[16][:2]
            left_shoulder = frame_keypoints[5][:2]  # 通常左肩索引为5
            right_shoulder = frame_keypoints[6][:2]  # 通常右肩索引为6
            torso_center = (
                (frame_keypoints[23][0] + frame_keypoints[24][0]) / 2,
                (frame_keypoints[23][1] + frame_keypoints[24][1]) / 2,
            )

            left_leg_angle = compute_angle(left_hip, left_knee, left_ankle)
            right_leg_angle = compute_angle(right_hip, right_knee, right_ankle)
            upper_body_angle = compute_angle(left_shoulder, torso_center, right_shoulder)

            angles.append([left_leg_angle, right_leg_angle, upper_body_angle])

        return angles

    def interpolate_missing_keypoints(self, keypoints):
        """
        使用线性插值法补全缺失的关键点。

        Args:
            keypoints (list): 每帧的关键点列表，每个关键点包含33个点，每个点有x, y, confidence。

        Returns:
            list: 补全后的关键点列表。
        """
        keypoints = np.array(keypoints)  # 转换为NumPy数组
        num_frames, num_keypoints, num_coords = keypoints.shape

        # 遍历每个关键点的x和y坐标
        for k in range(num_keypoints):
            for coord in range(2):  # 0: x, 1: y
                series = keypoints[:, k, coord]

                # 假设置信度低或值为0表示缺失
                missing = (series == 0) | np.isnan(series)

                if np.any(missing):
                    not_missing = ~missing
                    if np.sum(not_missing) < 2:
                        # 如果有效点少于2个，无法进行插值，全部设为0
                        series[:] = 0
                    else:
                        indices = np.arange(num_frames)
                        valid_indices = indices[not_missing]
                        valid_values = series[not_missing]

                        # 使用NumPy的插值函数进行线性插值
                        interpolated_values = np.interp(indices, valid_indices, valid_values)

                        # 替换缺失值
                        series[missing] = interpolated_values[missing]
                        keypoints[:, k, coord] = series

        return keypoints.tolist()