import cv2
import os
import glob
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import random

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

def get_image_shape(image_path):
    """
    Get the shape (channels, height, width) of a JPG image.

    Parameters:
        image_path (str): Path to the JPG image file.

    Returns:
        tuple: A tuple (channels, height, width).
    """
    with Image.open(image_path) as img:
        width, height = img.size
        mode = img.mode
        
        # Determine the number of channels based on the mode
        channels = {
            "1": 1,      # 1-bit pixels, black and white, stored with one pixel per byte
            "L": 1,      # 8-bit pixels, black and white
            "P": 1,      # 8-bit pixels, mapped to any other mode using a color palette
            "RGB": 3,    # 3x8-bit pixels, true color
            "RGBA": 4,   # 4x8-bit pixels, true color with transparency mask
            "CMYK": 4,   # 4x8-bit pixels, color separation
            "YCbCr": 3,  # 3x8-bit pixels, color video format
            "LAB": 3,    # 3x8-bit pixels, L*a*b color space
            "HSV": 3,    # 3x8-bit pixels, Hue, Saturation, Value color space
            "I": 1,      # 32-bit signed integer pixels
            "F": 1       # 32-bit floating point pixels
        }.get(mode, None)

        if channels is None:
            raise ValueError(f"Unsupported image mode: {mode}")

        return (channels, height, width)



# Extra functions for MM811
def convert_png_to_jpg(input_path: str):
    """
    Converts a PNG image to JPEG format and replaces the original image.

    Args:
        input_path (str): Path to the input PNG image.
    """
    try:
        # Ensure the file exists and has a .png extension
        if not os.path.isfile(input_path) or not input_path.lower().endswith(".png"):
            raise ValueError("Input file must be a valid PNG image.")

        # Open the PNG image
        image = Image.open(input_path).convert("RGB")  # Convert to RGB for JPEG format

        # Define the new file path with a .jpg extension
        output_path = os.path.splitext(input_path)[0] + ".jpg"

        # Save the image as JPEG
        image.save(output_path, "JPEG")

        # Remove the original PNG file
        os.remove(input_path)

        print(f"Image converted to JPEG and saved as {output_path}. Original PNG image deleted.")

    except Exception as e:
        print(f"Error converting image: {e}")

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

def display_image_and_capture_clicks(image_path, num_clicks):
    """
    Displays a JPG image and allows the user to click on it.
    Captures a specified number of clicks and returns their coordinates.

    Args:
        image_path (str): Path to the JPG image to display.
        num_clicks (int): The number of clicks to capture.

    Returns:
        list: List of (x, y) coordinates of the captured clicks.
    """
    # Open the image
    img = Image.open(image_path)
    
    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(f"Click {num_clicks} times on the image. Close the window when done.")

    # List to store coordinates
    coords = []

    # Event handler to capture clicks
    def onclick(event):
        if event.xdata is not None and event.ydata is not None:  # Check for valid click
            x, y = int(event.xdata), int(event.ydata)
            coords.append((x, y))
            print(f"Clicked at: ({x}, {y})")
            if len(coords) >= num_clicks:  # Stop after required number of clicks
                plt.close(fig)
    # Connect the event handler
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    # Show the image
    plt.show()
    # Disconnect the event handler
    fig.canvas.mpl_disconnect(cid)

    return coords

def display_image_and_capture_clicks_video(video_path, num_clicks):
    """
    Displays the first frame of a video and allows the user to click on it.
    Captures a specified number of clicks and returns their coordinates.

    Args:
        video_path (str): Path to the video file.
        num_clicks (int): The number of clicks to capture.

    Returns:
        list: List of (x, y) coordinates of the captured clicks.
    """
    # Extract the first frame from the video
    first_frame = get_first_frame(video_path)
    
    # Convert the BGR frame to RGB for displaying
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    
    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.imshow(first_frame_rgb)
    ax.set_title(f"Click {num_clicks} times on the image. Close the window when done.")

    # List to store coordinates
    coords = []

    # Event handler to capture clicks
    def onclick(event):
        if event.xdata is not None and event.ydata is not None:  # Check for valid click
            x, y = int(event.xdata), int(event.ydata)
            coords.append((x, y))
            print(f"Clicked at: ({x}, {y})")
            if len(coords) >= num_clicks:  # Stop after required number of clicks
                plt.close(fig)

    # Connect the event handler
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    # Show the image
    plt.show()
    
    # Disconnect the event handler
    fig.canvas.mpl_disconnect(cid)

    return coords

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

def png_to_video(frames_path, output_video_path, fps=30):
    """
    Convert a sequence of PNG files to a video.

    Parameters:
        frames_path (str): Path to the folder containing PNG frames.
        output_video_path (str): Path to save the output video file.
        fps (int, optional): Frames per second for the output video. Default is 30.
    
    Returns:
        None
    """
    # Get all PNG files using glob and sort naturally
    png_files = glob.glob(os.path.join(frames_path, "*.png"))
    png_files.sort(key=lambda f: int(''.join(filter(str.isdigit, os.path.basename(f)))) if ''.join(filter(str.isdigit, os.path.basename(f))) else 0)

    if not png_files:
        raise ValueError("No PNG files found in the specified frames path.")

    # Read the first image to get the video dimensions
    first_frame = cv2.imread(png_files[0])
    height, width, _ = first_frame.shape

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 format
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write frames to the video
    for frame_path in png_files:
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Skipping file {frame_path} as it couldn't be read.")
            continue
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()
    print(f"Video successfully created at: {output_video_path}")

def extract_masks(outpath, video_segments, frame_names):
    """
    Processes masks to make them white on a black background and saves them as images.
    
    Args:
        outpath (str): Directory to save the processed masks.
        The following 2 parameters are passed from SAM2 process_video_811
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
            output_file = os.path.join(outpath, frame_name)

            # Save the mask as a PNG image
            cv2.imwrite(output_file, binary_mask)
    
    print(f"Masks saved to {outpath}")

def extract_videos_frames(input_folder, output_folder):
    """
    Extracts frames from all .mp4 videos in the input folder and saves them as .jpg files
    in subfolders under the output folder, named after the video files.

    Args:
        input_folder (str): Path to the folder containing .mp4 video files.
        output_folder (str): Path to the folder where extracted frames will be stored.

    Returns:
        None
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".mp4"):
            video_path = os.path.join(input_folder, file_name)
            video_name = os.path.splitext(file_name)[0]
            frame_output_folder = os.path.join(output_folder, video_name)

            # Ensure a subfolder for the video frames exists
            os.makedirs(frame_output_folder, exist_ok=True)

            # Open the video file
            cap = cv2.VideoCapture(video_path)
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Save frame as zero-padded .jpg
                frame_filename = os.path.join(frame_output_folder, f"{frame_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                frame_count += 1

            cap.release()
            print(f"Extracted {frame_count} frames from {file_name} into {frame_output_folder}")

    print("Frame extraction completed.")

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

def enhance_images_in_folder(folder_path, alpha=1.5, beta=50):
    """
    Enhance and replace all .jpg images in a folder by adjusting brightness and contrast.
    
    Parameters:
    - folder_path (str): Path to the folder containing .jpg images.
    - alpha (float): Contrast control (1.0-3.0, default is 1.5).
    - beta (int): Brightness control (0-100, default is 50).
    """
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return
    
    # Get a list of all .jpg images in the folder
    image_files = glob.glob(os.path.join(folder_path, "*.jpg"))
    
    if not image_files:
        print(f"No .jpg images found in {folder_path}.")
        return
    
    # Process each image
    for image_file in image_files:
        try:
            # Read the image
            image = cv2.imread(image_file)
            if image is None:
                print(f"Failed to read {image_file}. Skipping.")
                continue
            
            # Enhance the image
            enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            
            # Replace the original image
            cv2.imwrite(image_file, enhanced_image)
            #print(f"Enhanced and replaced: {image_file}")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
    
    print("Enhancement complete for all images in the folder.")

def modify_color_style(input_path: str, output_path: str):
    """
    Simulates stronger lighting conditions for an input image and saves the result.

    Args:
        input_path (str): Path to the input image (e.g., JPEG, PNG).
        output_path (str): Path to save the output modified image.
    """
    try:
        # Open the input image
        image = Image.open(input_path).convert("RGB")

        # Apply moderate contrast adjustment
        contrast_factor = random.uniform(0.8, 1.3)  # Slightly lower or higher contrast
        contrast_enhancer = ImageEnhance.Contrast(image)
        image_contrasted = contrast_enhancer.enhance(contrast_factor)

        # Apply moderate color balance adjustment
        color_factor = random.uniform(0.85, 1.15)  # Slightly muted or enhanced colors
        color_enhancer = ImageEnhance.Color(image_contrasted)
        final_image = color_enhancer.enhance(color_factor)

        # Apply a warm or cool light tint, biased toward blue tones
        if random.choice([True, False]):  # 50% chance to apply a tint
            image_array = np.array(final_image, dtype=np.float32)
            
            # Adjust tint multipliers with a tendency toward blue
            warm_tint = np.array([
                random.uniform(0.9, 1.05),  # Red multiplier (slightly cooler red tones)
                random.uniform(0.9, 1.05),  # Green multiplier (slightly cooler green tones)
                random.uniform(1.0, 1.15)   # Blue multiplier (enhanced blue tones)
            ])
            
            image_array = np.clip(image_array * warm_tint, 0, 255).astype(np.uint8)
            final_image = Image.fromarray(image_array)

        # Save the modified image to the output path
        final_image.save(output_path, "JPEG")
        print(f"Image with blue-leaning random color style saved to {output_path}")

    except Exception as e:
        print(f"Error processing the image: {e}")

def apply_random_color_style(image, brightness_factor, contrast_factor, color_factor, warm_tint):
    """
    Applies a consistent random color style to a single frame.
    """
    # Apply brightness adjustment
    brightness_enhancer = ImageEnhance.Brightness(image)
    image_brightened = brightness_enhancer.enhance(brightness_factor)

    # Apply contrast adjustment
    contrast_enhancer = ImageEnhance.Contrast(image_brightened)
    image_contrasted = contrast_enhancer.enhance(contrast_factor)

    # Apply color balance adjustment
    color_enhancer = ImageEnhance.Color(image_contrasted)
    final_image = color_enhancer.enhance(color_factor)

    # Apply warm/cool tint
    image_array = np.array(final_image, dtype=np.float32)
    image_array = np.clip(image_array * warm_tint, 0, 255).astype(np.uint8)
    final_image = Image.fromarray(image_array)

    return final_image

def convert_jpgs_to_video_with_color_style(input_folder, output_video_path, fps=30):
    """
    Converts a sequence of JPG images into a video with a consistent random color style applied to all frames.

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

    # Generate random factors for the consistent color style
    brightness_factor = np.random.uniform(0.8, 1.2)
    contrast_factor = np.random.uniform(0.8, 1.3)
    color_factor = np.random.uniform(0.85, 1.15)
    warm_tint = np.array([np.random.uniform(0.95, 1.1),  # Red multiplier
                          np.random.uniform(0.95, 1.1),  # Green multiplier
                          np.random.uniform(0.9, 1.05)])  # Blue multiplier

    # Read the first image to get frame dimensions
    first_frame_path = os.path.join(input_folder, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    height, width, _ = first_frame.shape

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print("Converting images to video with consistent random color style...")
    for frame_file in frame_files:
        frame_path = os.path.join(input_folder, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Skipping invalid frame: {frame_file}")
            continue

        # Convert OpenCV frame (BGR) to PIL image (RGB)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Apply the consistent random color style
        styled_image = apply_random_color_style(image, brightness_factor, contrast_factor, color_factor, warm_tint)

        # Convert back to OpenCV format (BGR) and write to video
        styled_frame = cv2.cvtColor(np.array(styled_image), cv2.COLOR_RGB2BGR)
        video_writer.write(styled_frame)

    video_writer.release()  # Finalize the video
    print(f"Video with random color style saved at: {output_video_path}")

#---------------#
# Testing Field #
#---------------#
v1_raw_frams = "../data/fall/raw_frames/1"
v1_raw_video = "../data/raw_videos/1.mp4"
v1_points = "../data/points/1.txt"
v1_point_frames = "../data/point_frames/1"
v1_point_video = "../data/point_videos/1.mp4"
example_image = v1_raw_frams+"/0000.jpg"
#print(get_image_shape("../data/train/not_fall/raw_frames/4/0000.jpg"))
# rename frames
#rename_and_convert_frames(v1_raw_frams+"/1", v1_raw_frams)
v_raw_frams = "../data/train/not_fall/raw_frames/raw"
v_raw_frams_jpg = "../data/train/not_fall/raw_frames/4"
v_raw_video = "../data/train/not_fall/raw_videos/4.mp4"

'''
rename_and_convert_frames(v_raw_frams,v_raw_frams_jpg)
enhance_images_in_folder(v_raw_frams_jpg)
convert_jpgs_to_video(v_raw_frams_jpg, v_raw_video)
'''
#convert_jpgs_to_video_with_color_style(v_raw_frams_jpg, v_raw_video)
# Jpg to video
#convert_jpgs_to_video(v1_raw_frams,v1_raw_video)

# draw points on frames

#draw_points_on_frames(v1_points, v1_raw_frams, v1_point_frames)
#convert_jpgs_to_video(v1_point_frames,v1_point_video)


#coords = display_image_and_capture_clicks_video(v_raw_video, 2)
#print(coords)

#shape = get_image_shape(example_image)
#print("Image shape (channels, height, width):", shape)

#modify_color_style("1.jpg","2.jpg")
#apply_blue_style("1.jpg","2.jpg")

v_raw_videos = "../data/test/not_fall/raw_videos"
output_frames = "../data/test/not_fall/raw_frames"

#extract_videos_frames(v_raw_videos, output_frames)
#convert_jpgs_to_video("../data/test/not_fall/masks/1","../data/test/not_fall/mask_videos/1.mp4")