import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_video_predictor
import time
from frame_manager import get_video_properties, reconstruct_video, extract_mask_centers, draw_points_on_frames, extract_masks, display_image_and_capture_clicks, display_image_and_capture_clicks_video


# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

# Setups
checkpoint = "../checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = os.path.abspath("../configs/sam2.1_hiera_t.yaml")
predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)

#np.random.seed(3)

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def process_video(video):
    frames_dir = "../frames/" + video
    output_frames_dir = "../output_frames"
    output_dir = "../output_videos"
    os.makedirs(output_dir, exist_ok=True)
    # Load and sort frame names
    frame_names = [
        p for p in os.listdir(frames_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0].split('_')[-1]))

    print("Start Processing Frames...")
    frame_processing_start = time.time()

    inference_state = predictor.init_state(video_path=frames_dir, offload_video_to_cpu=True, async_loading_frames=True)

    # Define prompt points for object tracking
    ann_frame_idx = 0 
    ann_obj_id = 1
    points = np.array([[800, 400], [780, 250]], dtype=np.float32)
    labels = np.array([1, 1], np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    # Propagate prompt points through video
    video_segments = {}
    segmented_frames = {}

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    frame_processing_end = time.time()
    frame_processing_time = frame_processing_end - frame_processing_start
    print(" ")
    print("Frame Masks Extracted.")
    print(f"Frame Processing Time: {frame_processing_time:.2f} seconds")

    # Video Reconstruction
    print("Reconstructing video...")
    video_reconstruction_start = time.time()
    video_source = "Video" + video + ".mp4"
    video_target = "Video" + video + "_output.mp4"
    video1_path = os.path.join("..", "Videos", video_source)
    properties_v1 = get_video_properties(video1_path)
    output_path = os.path.abspath(os.path.join(output_dir, video_target))
    reconstruct_video(output_path, frames_dir, properties_v1, frame_names, video_segments)
    video_reconstruction_end = time.time()
    video_reconstruction_time = video_reconstruction_end - video_reconstruction_start
    print(f"Video Reconstruction Time: {video_reconstruction_time:.2f} seconds")
    print(f"Reconstructed video finished. Check '{output_path}'")

    # Time Count
    total_processing_time = frame_processing_time + video_reconstruction_time
    print(f"Total Video Processing Time: {total_processing_time:.2f} seconds")

    predictor.reset_state(inference_state)

def process_video2(video):
    frames_dir = "../frames/" + video
    output_frames_dir = "../output_frames"
    output_dir = "../output_videos"
    os.makedirs(output_dir, exist_ok=True)

    # Load and sort frame names
    frame_names = [
        p for p in os.listdir(frames_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0].split('_')[-1]))

    split_index = 1386
    video_segments = {}
    print("Start Processing Frames...")
    frame_processing_start = time.time()

    # Define prompt points for each segment
    # Seperate frames into 2 parts since the object blocking cause poor performance, thus need to manually set new prompt points 
    points_part1 = np.array([[379, 395], [428, 397]], dtype=np.float32)
    points_part2 = np.array([[885, 300], [911, 299]], dtype=np.float32)
    labels = np.array([1, 1], np.int32)

    # Process frames 0 ~ 1389 with points_part1
    inference_state = predictor.init_state(video_path=frames_dir, offload_video_to_cpu=True, async_loading_frames=True)
    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=points_part1,
        labels=labels,
    )

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        if out_frame_idx < split_index:
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
    predictor.reset_state(inference_state)

    # Process frames 1389 ~ end with points_part2
    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=1386,
        obj_id=1,
        points=points_part2,
        labels=labels,
    )

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        if out_frame_idx >= split_index:
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

    # Video Reconstruction
    print(" ")
    print("Frame Masks Extracted.")
    frame_processing_end = time.time()
    frame_processing_time = frame_processing_end - frame_processing_start
    print(f"Frame Processing Time: {frame_processing_time:.2f} seconds")

    video_source = "Video" + video + ".mp4"
    video_target = "Video" + video + "_output.mp4"
    video1_path = os.path.join("..", "Videos", video_source)
    properties_v1 = get_video_properties(video1_path)
    output_path = os.path.abspath(os.path.join(output_dir, video_target))

    print("Reconstructing video...")
    video_reconstruction_start = time.time()
    reconstruct_video(output_path, frames_dir, properties_v1, frame_names, video_segments)
    video_reconstruction_end = time.time()
    video_reconstruction_time = video_reconstruction_end - video_reconstruction_start
    print(f"Video Reconstruction Time: {video_reconstruction_time:.2f} seconds")
    print(f"Reconstructed video finished. Check '{output_path}'")
    
    # Time Count
    total_processing_time = video_reconstruction_time + frame_processing_time
    print(f"Total Video Processing Time: {total_processing_time:.2f} seconds")
    predictor.reset_state(inference_state)

'''
def process_video_811(video, points):
    """
    Processes a video by using object tracking and segmentation techniques.

    Args:
        video (str): The name of the video to process (used for folder structure).
        points (numpy.ndarray): Array of points (coordinates) for object tracking.

    Returns:
        None
    """
    frames_dir = "../data/train/not_fall/raw_frames/" + video
    #frames_dir = "../data/train/not_fall/raw_frames/raw"
    output_dir = "../data/train/not_fall/mask_videos"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and sort frame names
    frame_names = [
        p for p in os.listdir(frames_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0].split('_')[-1]))

    print("Start Processing Frames...")
    frame_processing_start = time.time()

    inference_state = predictor.init_state(video_path=frames_dir, offload_video_to_cpu=True, async_loading_frames=True)

    # Define labels corresponding to the provided points
    labels = np.ones(len(points), dtype=np.int32)  # Assuming all points are foreground points
    
    # Annotate the initial frame with the given points
    ann_frame_idx = 0  # Annotated frame index
    ann_obj_id = 1  # ID for the object being tracked
    
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    # Propagate prompt points through the video
    video_segments = {}
    segmented_frames = {}

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    frame_processing_end = time.time()
    frame_processing_time = frame_processing_end - frame_processing_start
    print(" ")
    print("Frame Masks Extracted.")
    print(f"Frame Processing Time: {frame_processing_time:.2f} seconds")

    # Video Reconstruction + Frames processing
    print("Reconstructing video...")
    video_reconstruction_start = time.time()
    video_source = video + ".mp4"
    video_target = video + ".mp4"
    video_path = os.path.join("..", "data","train","not_fall", "raw_videos", video_source)
    properties_v = get_video_properties(video_path)
    output_path = os.path.abspath(os.path.join(output_dir, video_target))
    
    points_file = "../data/train/not_fall/points/"+video+".txt"
    point_frames = "../data/train/not_fall/point_frames/"+video
    point_mask_video = "../data/train/not_fall/point_mask_videos/"+video+".mp4"
    bw_masks = "../data/train/not_fall/masks/"+video
    # Get mass centers
    #extract_mask_centers(points_file, video_segments, frame_names)
    # Get mask images
    #extract_masks(bw_masks, video_segments, frame_names)
    # Genertae point frames
    #draw_points_on_frames(points_file,frames_dir,point_frames)
    # Generate masked video
    reconstruct_video(output_path, frames_dir, properties_v, frame_names, video_segments)
    # Generate masked video with points
    #reconstruct_video(point_mask_video, point_frames, properties_v, frame_names, video_segments)
    
    video_reconstruction_end = time.time()
    video_reconstruction_time = video_reconstruction_end - video_reconstruction_start
    print(f"Video Reconstruction Time: {video_reconstruction_time:.2f} seconds")
    print(f"Reconstructed video finished. Check '{output_path}'")

    # Time Count
    total_processing_time = frame_processing_time + video_reconstruction_time
    print(f"Total Video Processing Time: {total_processing_time:.2f} seconds")

    predictor.reset_state(inference_state)
'''

def process_video_811(video, points, base_path, reset_points=False, num_clicks=2):
    """
    Processes a video by using object tracking and segmentation techniques.

    Args:
        video (str): The name of the video to process (used for folder structure).
        points (numpy.ndarray): Array of points (coordinates) for object tracking.
        base_path (str): Base path to prepend to all relative paths.
        reset_points (bool): If True, recapture points from the first frame of the video.
        num_clicks (int): Number of clicks to capture if reset_points is True.
    Returns:
        None
    """
    # Define directories using formatted strings
    frames_dir = f"{base_path}/raw_frames/{video}"
    raw_video = f"{base_path}/raw_videos/{video}.mp4"
    print(raw_video)
    output_dir = f"{base_path}/mask_videos"
    os.makedirs(output_dir, exist_ok=True)

    # Capture click points from the first frame of the video
    if reset_points:
        points = display_image_and_capture_clicks_video(raw_video, num_clicks)
    
    # Load and sort frame names
    frame_names = [
        p for p in os.listdir(frames_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0].split('_')[-1]))

    print("Start Processing Frames...")
    frame_processing_start = time.time()

    inference_state = predictor.init_state(video_path=frames_dir, offload_video_to_cpu=True, async_loading_frames=True)

    # Define labels corresponding to the provided points
    labels = np.ones(len(points), dtype=np.int32)  # Assuming all points are foreground points
    
    # Annotate the initial frame with the given points
    ann_frame_idx = 0  # Annotated frame index
    ann_obj_id = 1  # ID for the object being tracked
    
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    # Propagate prompt points through the video
    video_segments = {}
    segmented_frames = {}

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    frame_processing_end = time.time()
    frame_processing_time = frame_processing_end - frame_processing_start
    print(" ")
    print("Frame Masks Extracted.")
    print(f"Frame Processing Time: {frame_processing_time:.2f} seconds")

    # Video Reconstruction + Frames processing
    print("Reconstructing video...")
    video_reconstruction_start = time.time()
    video_source = f"{video}.mp4"
    video_target = f"{video}.mp4"
    video_path = f"{base_path}/raw_videos/{video_source}"
    properties_v = get_video_properties(video_path)
    output_path = f"{output_dir}/{video_target}"
    
    points_file = f"{base_path}/points/{video}.txt"
    point_frames = f"{base_path}/point_frames/{video}"
    point_mask_video = f"{base_path}/point_mask_videos/{video}.mp4"
    bw_masks = f"{base_path}/masks/{video}"
    
    # Get mass centers
    #extract_mask_centers(points_file, video_segments, frame_names)
    # Get mask images
    extract_masks(bw_masks, video_segments, frame_names)
    # Generate point frames
    #draw_points_on_frames(points_file, frames_dir, point_frames)
    # Generate masked video
    #reconstruct_video(output_path, frames_dir, properties_v, frame_names, video_segments)
    # Generate masked video with points
    #reconstruct_video(point_mask_video, point_frames, properties_v, frame_names, video_segments)
    
    video_reconstruction_end = time.time()
    video_reconstruction_time = video_reconstruction_end - video_reconstruction_start
    print(f"Video Reconstruction Time: {video_reconstruction_time:.2f} seconds")
    print(f"Reconstructed video finished. Check '{output_path}'")

    # Time Count
    total_processing_time = frame_processing_time + video_reconstruction_time
    print(f"Total Video Processing Time: {total_processing_time:.2f} seconds")

    predictor.reset_state(inference_state)

def process_videos_in_folder(folder_path, num_clicks):
    """
    Loops through all videos in a given folder, captures click points from the first frame
    of each video, and processes them with `process_video_811`.

    Args:
        folder_path (str): Path to the folder containing video files.
        num_clicks (int): Number of clicks to capture from each video.

    Returns:
        None
    """
    # Get all video files in the folder
    video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]

    # Extract the based path that will be passed to process_video_811
    folder_parts = folder_path.split('/')
    based_path = '/'.join(folder_parts[:-1])

    for video_file in video_files:
        # Extract the video name (e.g., "1" from "1.mp4")
        video_name = os.path.splitext(video_file)[0]

        # Full path to the video
        video_path = os.path.join(folder_path, video_file)

        print(f"Processing video: {video_file}")

        # Process the video with the captured points
        process_video_811(video_name, None, based_path, True, num_clicks)
        print(f"Finished processing video: {video_file}\n")


'''
points = np.array([[465, 201], [424, 360]], dtype=np.float32)
process_video_811("1", points)
'''
raw_videos = "../data/raw_videos"
v_raw_video = "../data/train/not_fall/raw_videos/4.mp4"
points = np.array([[118, 185], [125, 248]], dtype=np.float32)
base_path = "../data/train/not_fall"
#process_video_811("4", points,base_path)

#process_videos_in_folder("../data/test/fall/raw_videos",2)