import yaml 
import glob
import numpy as np  
import os  
import plot_utils
from torchvision.io import decode_image
from torchvision.utils import draw_segmentation_masks
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd 

def read_config_yaml(config_path):
    """
    Reads in configuration YAML file and converts it
    to a dictionary. 

    Parameters
    ----------
    config_path : str
        The full path to the configuration file 

    Returns
    -------
    dict
        Dictionary of configuration parameters 
    """

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)  # safe_load avoids arbitrary code execution

    return config

def filter_annotations(annotations_file, fps, SAM2_start):

    ####### READ NPY FILE CONTAINING GUI ANNOTATIONS###########
    annotations = np.load(annotations_file, allow_pickle=True)

    # Filter out "Fish_Fam" key and modify "Frame" values
    annotations_filtered = []
    for ann in annotations:
        filtered_ann = {}
        for key, value in ann.items():
            if key != "Fish_Fam":
                if key == "Frame":
                    frame_value = (value - SAM2_start - 1) / (fps / 3)
                    #Check if frame_value is a decimal
                    if not frame_value.is_integer():
                        rounded_frame_value = round(frame_value)
                        print(f"Warning: Frame value {frame_value:.2f} was rounded to {rounded_frame_value}")
                        frame_value = rounded_frame_value
                    filtered_ann[key] = int(frame_value)
                else:
                    filtered_ann[key] = value
        annotations_filtered.append(filtered_ann)

    return annotations_filtered    

def filter_annotations_v2(annotations_file=None, fps=None, SAM2_start=None):
    """
    TODO: fill in the details here 

    """

    # Read in npy file with annotations and convert it to a Pandas df 
    annotations = np.load(annotations_file, allow_pickle=True)

    df = pd.DataFrame(list(annotations))

    if 'Fish_Fam' in df.columns:
        # Drop Fish_Fam column 
        df = df.drop(columns=['Fish_Fam'])

    # Correct annotation frame value, so it coincides with video frame value
    df['Frame'] = (df['Frame'] - SAM2_start - 1) / (fps / 3)

    # Round and convert Frame value to an integer 
    # TODO: do we need to provide a warning that a rounding was necessary? 
    df['Frame'] = round(df['Frame']).astype(int)

    return df 

def get_frame_chunks_df(df=None):
    """
    TODO: fill in details 

    modifies df in place (makes fishLabel the index and drops clickType of 3 and 4)
    creates `fish_frame_chunks`
    """

    # Modify the incoming df so it has fishLabel as its index
    # TODO: should we do this inplace? 
    df = df.set_index('fishLabel')

    # For each fishLabel get Frames where the fish enters the scene 
    enter_frame = df[df['clickType'] == 3]['Frame']

    # TODO: this is temporary so we can test cases where duplication happens
    if enter_frame.index.duplicated().any():
        raise RuntimeError("Provided annotations have more than 1 enter frame for a fish label!")
    
    # For each fishLabel get Frames where the fish exits the scene 
    exit_frame = df[df['clickType'] == 4]['Frame']

    # TODO: this is temporary so we can test cases where duplication happens
    if exit_frame.index.duplicated().any():
        raise RuntimeError("Provided annotations have more than 1 exit frame for a fish label!")

    # Concatenate columns to improve ease of use later
    fish_frame_chunks = pd.concat([enter_frame, exit_frame], axis=1)
    fish_frame_chunks.columns = ['EnterFrame', 'ExitFrame']

    # Drop rows that have clickType of 3 or 4 from input df 
    # TODO: should we do this inplace? 
    df = df[~df['clickType'].isin([3, 4])]

    return fish_frame_chunks, df

def get_jpg_paths(jpg_dir):
    """
    Compiles a list of paths for all JPGs in the provided directory. 

    Parameters
    ----------
    jpg_dir : str
        The full path to the directory containing JPGs

    Returns
    -------
    list
        List of sorted JPG paths
    """

    # Grab all files with extensions .jpg, .jpeg, .JPG, .JPEG in jpg_dir
    jpg_files = glob.glob(os.path.join(jpg_dir, '*.[jJ][pP][gG]'))
    jpeg_files = glob.glob(os.path.join(jpg_dir, '*.[jJ][pP][eE][gG]'))

    # TODO: make these Path objects 
    jpg_paths = jpg_files + jpeg_files

    return sorted(jpg_paths)

def draw_and_save_frame_seg(bool_masks, img_save_dir, frame_paths, out_frame_idx, out_obj_ids, colors, 
                            font_size=75, font_color="red", alpha=0.6):
    """
    Draws segmentation masks on top of the frame and saves 
    the generated image to `img_save_dir`. 

    Parameters
    ----------
    bool_masks : Tensor of bools
        A tensor of shape (number of masks, frame pixel height, frame pixel width)
        representing the generated segmentation masks
    img_save_dir : str
        The path where we want to save the generated image
    frame_paths : list of str
        A list of JPG paths representing the frames 
    out_frame_idx : int
        Index representing the frame we predicted the masks for 
    out_obj_ids : list of int
        A list of integers representing the ids for each mask
    colors : list of tuples of ints
        A list of tuples representing RGB colors for each segmentation mask
    font_size : int
        Font size for drawn object IDs
    font_color : str
        Color of font for the drawn object IDs
    alpha : float 
        Alpha value for the segmentation masks 

    Returns
    -------
    list
        List of sorted JPG paths
    """    

    # Draw each mask on top of image representing the frame
    image_w_seg = decode_image(frame_paths[out_frame_idx])
    for i in range(bool_masks.shape[0]):

        # Only draw masks that contain True values 
        if bool_masks[i].any():
            image_w_seg = draw_segmentation_masks(image_w_seg, bool_masks[i], colors=colors[i], alpha=alpha)

    # Convert image with drawn segmentation masks to PIL Image
    to_pil = transforms.ToPILImage()
    img_pil = to_pil(image_w_seg)

    # Draw text annotations
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.load_default(size=font_size)

    # Compute centroids for all masks so we can place the object ID on the centroid 
    centroids = [plot_utils.get_centroid(mask) for mask in bool_masks]

    # Draw the object ID at the centroid
    for centroid, label in zip(centroids, out_obj_ids):
        if centroid:
            draw.text(centroid, str(label), fill=font_color, font=font)

    # Get frame name using the stem of the frame JPG
    frame_id = Path(frame_paths[out_frame_idx]).stem

    # Save the final image
    img_pil.save(img_save_dir + f"/{frame_id}.jpg")

def draw_and_save_frame_seg_v2(bool_masks, img_save_dir, frame_paths, out_frame_idx, out_obj_ids, colors, 
                               font_size=75, font_color="red", alpha=0.6):
    """
    Draws segmentation masks on top of the frame and saves 
    the generated image to `img_save_dir`. 

    Parameters
    ----------
    bool_masks : Tensor of bools
        A tensor of shape (number of masks, frame pixel height, frame pixel width)
        representing the generated segmentation masks
    img_save_dir : str
        The path where we want to save the generated image
    frame_paths : list of str
        A list of JPG paths representing the frames 
    out_frame_idx : int
        Index representing the frame we predicted the masks for 
    out_obj_ids : list of int
        A list of integers representing the ids for each mask
    colors : list of tuples of ints
        A list of tuples representing RGB colors for each segmentation mask
    font_size : int
        Font size for drawn object IDs
    font_color : str
        Color of font for the drawn object IDs
    alpha : float 
        Alpha value for the segmentation masks 

    Returns
    -------
    list
        List of sorted JPG paths

    TODO: make sure docstrings are good 
    """    

    # Get frame name using the stem of the frame JPG
    frame_id = Path(frame_paths[out_frame_idx]).stem

    # Draw each mask on top of image representing the frame
    # image_w_seg = decode_image(frame_paths[out_frame_idx])
    image_w_seg = decode_image(img_save_dir + f"/{frame_id}.jpg") # TODO: clean 
    for i in range(bool_masks.shape[0]):

        # Only draw masks that contain True values 
        if bool_masks[i].any():
            image_w_seg = draw_segmentation_masks(image_w_seg, bool_masks[i], colors=colors[out_obj_ids[i]], alpha=alpha)

    # Convert image with drawn segmentation masks to PIL Image
    to_pil = transforms.ToPILImage()
    img_pil = to_pil(image_w_seg)

    # Draw text annotations
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.load_default(size=font_size)

    # Compute centroids for all masks so we can place the object ID on the centroid 
    centroids = [plot_utils.get_centroid(mask) for mask in bool_masks]

    # Draw the object ID at the centroid
    for centroid, label in zip(centroids, out_obj_ids):
        if centroid:
            draw.text(centroid, str(label), fill=font_color, font=font)

    # Save the final image
    img_pil.save(img_save_dir + f"/{frame_id}.jpg")

def write_output_video(masked_imgs_dir, video_file, video_fps, video_frame_size):
    """
    Compiles the JPGs in `masked_imgs_dir` into an MP4. 

    Parameters
    ----------
    masked_imgs_dir : str 
        The full path to the directory containing the JPGs we 
        will use to create the video 
    video_file : str
        The name of the video file to be created 
    video_fps : int
        The frames per second for the video 
    video_frame_size : list or tuple of ints
        Specifies the frame size for the video, with the first 
        element representing the width and the second corresponding
        to the height
    """

    masked_img_paths = get_jpg_paths(masked_imgs_dir)

    if not masked_img_paths:
        print(f"No images found in the path: {masked_imgs_dir}.")
        return

    # Set the width and height of the video 
    width = video_frame_size[0]
    height = video_frame_size[1]
    
    # Define the video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_file, fourcc, video_fps, video_frame_size)

    # Define font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, width / 1000)  # Adjust based on image size
    font_thickness = 2
    text_color = (255, 255, 255)  # White text

    # Write each image to the video with modifications
    for frame_idx, img_path in tqdm(enumerate(masked_img_paths), total=len(masked_img_paths)):

        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}, skipping.")
            continue

        # Get original image dimensions (before resizing)
        orig_height, orig_width = img.shape[:2]
        
        # Resize the image
        img = cv2.resize(img, video_frame_size)

        # Create a matplotlib figure
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

        # Display the image
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Set title with frame number
        ax.set_title(f"Frame {frame_idx + 1}/{len(masked_img_paths)}", fontsize=16)

        # Set tick marks based on the original image dimensions
        ax.set_xticks(np.linspace(0, width, num=10))  # 10 evenly spaced ticks
        ax.set_xticklabels(np.linspace(0, orig_width, num=10, dtype=int))  # Map to original width
        ax.set_yticks(np.linspace(0, height, num=10))
        ax.set_yticklabels(np.linspace(0, orig_height, num=10, dtype=int))  # Map to original height

        # Set axis labels
        ax.set_xlabel("Pixel value")
        ax.set_ylabel("Pixel value")

        # Remove tick labels to keep only marks
        ax.tick_params(axis='both', labelsize=10, color='black')

        # Convert Matplotlib figure to an image
        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())

        # Convert RGBA to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        # Write the frame to the video
        video.write(frame)

        # Close the figure to save memory
        plt.close(fig)
    
    # Release the video writer
    video.release()
