import cv2
import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Setups
#video_path = 'Video1.mp4'
#video_path = os.path.join("..", "Videos", "Video1.mp4")

# Functions
def get_first_frame(video_path):
    video = cv2.VideoCapture(video_path)
    ret, first_frame = video.read()
    video.release()
    return first_frame

def get_video_properties(video_path):
    video = cv2.VideoCapture(video_path)
    ret, first_frame = video.read()

    fps = video.get(cv2.CAP_PROP_FPS)
    height, width, channels = first_frame.shape

    video.release()

    video_properties = [width, height, channels, fps]
    return video_properties

def extract_frames(video):
    video_path = "../Videos/Video" + str(video) + ".mp4"
    output_folder = "../frames/" + str(video)
    #output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frames')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video = cv2.VideoCapture(video_path)
    
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frame_filename = os.path.join(output_folder, f'{len(os.listdir(output_folder)):04d}.jpg')
        cv2.imwrite(frame_filename, frame)

    video.release()
    print(f"Frames extracted to '{output_folder}'")

# reconstruct from memory masks
def reconstruct_video(output_path, frames_dir, properties, frame_names, video_segments):
    width, height, layers, fps = properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_idx, frame_name in enumerate(frame_names):
        frame_path = os.path.join(frames_dir, frame_name)
        frame = cv2.imread(frame_path)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        for out_obj_id, out_mask in video_segments[frame_idx].items():
            mask = (out_mask > 0).astype(np.uint8) * 255
            color_mask = cv2.merge([mask, mask, mask])
            frame = np.where(color_mask == 255, frame, gray_frame)
        video.write(frame[0])
    video.release()
    print("Video Reconstruction Done.")

def rename_and_convert_frames(input_folder_path, output_folder_path):
    """
    Renames and converts image frames in the given folder to sequentially numbered
    filenames in JPG format (e.g., 0000.jpg to 9999.jpg) and saves them in the output folder.

    Args:
        input_folder_path (str): Path to the folder containing the PNG files.
        output_folder_path (str): Path to the folder where converted files will be saved.

    Returns:
        None
    """
    # Ensure the output folder exists
    os.makedirs(output_folder_path, exist_ok=True)

    # List all PNG files in the input folder
    files = [f for f in os.listdir(input_folder_path) if f.endswith('.png')]
    files.sort()  # Sort files alphabetically

    for index, file_name in enumerate(files):
        # Create new file name in the format 0000.jpg
        new_file_name = f"{index:04d}.jpg"
        
        # Full paths for the original and new files
        original_file_path = os.path.join(input_folder_path, file_name)
        new_file_path = os.path.join(output_folder_path, new_file_name)
        
        # Convert PNG to JPG and save with new name in the output folder
        with Image.open(original_file_path) as img:
            rgb_img = img.convert('RGB')  # Convert to RGB for JPG format
            rgb_img.save(new_file_path, "JPEG")




# Extra functions for MM811

def display_image_and_capture_clicks(image_path):
    """
    Displays a JPG image and allows the user to click on it. 
    Prints the coordinates of each click.

    Args:
        image_path (str): Path to the JPG image to display.

    Returns:
        None
    """
    # Open the image
    img = Image.open(image_path)
    
    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title("Click on the image to get coordinates. Close the window when done.")

    # List to store coordinates
    coords = []

    # Event handler to capture clicks
    def onclick(event):
        if event.xdata is not None and event.ydata is not None:  # Check for valid click
            x, y = int(event.xdata), int(event.ydata)
            coords.append((x, y))
            print(f"Clicked at: ({x}, {y})")

    # Connect the event handler
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    # Show the image
    plt.show()
    
    # Disconnect after window closes
    fig.canvas.mpl_disconnect(cid)

def convert_jpgs_to_video(input_folder, output_video_path, fps=30):
    """
    Converts a sequence of JPG images into a video, sorted by filename.

    Args:
        input_folder (str): Path to the folder containing the JPG images.
        output_video_path (str): Path to save the output video.
        fps (int): Frames per second for the output video.

    Returns:
        None
    """
    # Get a list of all JPG files in the folder
    frame_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg')]
    frame_files.sort()  # Sort files by name

    if not frame_files:
        print("No JPG files found in the specified folder.")
        return

    # Read the first image to get frame dimensions
    first_frame_path = os.path.join(input_folder, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    height, width, _ = first_frame.shape

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print("Converting images to video...")
    for frame_file in frame_files:
        frame_path = os.path.join(input_folder, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Skipping invalid frame: {frame_file}")
            continue
        video_writer.write(frame)  # Write the frame to the video

    video_writer.release()  # Finalize the video
    print(f"Video saved at: {output_video_path}")

def extract_masks(outpath, video_segments, frame_names):
    """
    Processes masks to make them white on a black background and saves them as images.
    
    Args:
        outpath (str): Directory to save the processed masks.
        video_segments (list): List of dictionaries where each dictionary contains object ID and mask for a frame.
        frame_names (list): List of frame names corresponding to the frames in video_segments.
    """
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    for frame_idx, frame_name in enumerate(frame_names):
        frame_segments = video_segments[frame_idx]
        for obj_id, mask in frame_segments.items():
            # Ensure the mask is 2D by removing singleton dimensions
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)  # Remove the first dimension
            elif mask.ndim != 2:
                raise ValueError(f"Unexpected mask shape: {mask.shape}")

            # Create a white-on-black mask
            binary_mask = (mask > 0).astype(np.uint8) * 255

            # Construct the output file path
            output_file = os.path.join(outpath, f"{frame_name}_obj_{obj_id}.png")

            # Save the mask as a PNG image
            cv2.imwrite(output_file, binary_mask)
    
    print(f"Masks saved to {outpath}")

def extract_mask_centers(output_txt_path, video_segments, frame_names):
    """
    Extracts the center of mass for each mask in each frame and saves to a text file.
    
    Args:
        output_txt_path (str): Path to save the text file containing mask centers.
        video_segments (list): List of dictionaries where each dictionary contains object ID and mask for a frame.
        frame_names (list): List of frame names corresponding to the frames in video_segments.
    """
    rows = []

    for frame_idx, frame_name in enumerate(frame_names):
        frame_segments = video_segments[frame_idx]
        for obj_id, mask in frame_segments.items():
            # Ensure the mask is 2D by removing singleton dimensions
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)  # Remove the first dimension
            elif mask.ndim != 2:
                raise ValueError(f"Unexpected mask shape: {mask.shape}")

            # Calculate the center of mass of the mask
            y_coords, x_coords = np.nonzero(mask > 0)
            if len(x_coords) > 0:
                cx = np.mean(x_coords)
                cy = np.mean(y_coords)
                # Append a row with the format: FrameName x_center y_center
                rows.append(f"{frame_name} {cx:.2f} {cy:.2f}")

    # Write the rows to a text file
    with open(output_txt_path, 'w') as f:
        f.write("\n".join(rows))
    
    print(f"Mask centers saved to {output_txt_path}")

def draw_points_on_frames(input_txt_path, input_frames_path, output_frames_path, point_color=(0, 0, 255), point_radius=5):
    """
    Draws points on frames based on the coordinates in the text file and saves updated frames to output directory.
    
    Args:
        input_txt_path (str): Path to the text file containing frame names and coordinates.
        input_frames_path (str): Path to the directory containing the input frames.
        output_frames_path (str): Path to the directory where updated frames will be saved.
        point_color (tuple): BGR color of the point to be drawn.
        point_radius (int): Radius of the point to be drawn.
    """
    # Ensure the output directory exists
    os.makedirs(output_frames_path, exist_ok=True)

    # Read the coordinates from the text file
    with open(input_txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 3:
            raise ValueError(f"Invalid line format: {line.strip()}")

        frame_name, x_coord, y_coord = parts
        x_coord, y_coord = float(x_coord), float(y_coord)

        # Read the frame
        input_frame_path = os.path.join(input_frames_path, frame_name)
        if not os.path.exists(input_frame_path):
            print(f"Frame not found: {input_frame_path}")
            continue

        frame = cv2.imread(input_frame_path)
        if frame is None:
            print(f"Error reading frame: {input_frame_path}")
            continue

        # Draw the point on the frame
        point = (int(x_coord), int(y_coord))
        cv2.circle(frame, point, point_radius, point_color, -1)

        # Save the updated frame to the output directory
        output_frame_path = os.path.join(output_frames_path, frame_name)
        cv2.imwrite(output_frame_path, frame)

    print(f"Updated frames with points saved to {output_frames_path}")


#---------------#
# Testing Field #
#---------------#

# Jpg to video
v1_raw_frams = "../data/raw_frames/1"
v1_raw_video = "../data/raw_videos/1.mp4"
#convert_jpgs_to_video(v1_raw_frams,v1_raw_video)

# draw points on frames
v1_points = "../data/points/1.txt"
v1_point_frames = "../data/point_frames/1"
v1_point_video = "../data/point_videos/1.mp4"
#draw_points_on_frames(v1_points, v1_raw_frams, v1_point_frames)
#convert_jpgs_to_video(v1_point_frames,v1_point_video)