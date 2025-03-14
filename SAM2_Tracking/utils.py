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

def adjust_annotations(annotations_file=None, fps=None, SAM2_start=None, 
                       df_columns=None, frame_col_name=None):
    """
    Reads in a dictionary of annotations and converts it to a Pandas 
    DataFrame. Additionally, adjusts provided annotations so the 
    frame value aligns with SAM2. Adjustment is dictated by the 
    formula: `(frame_number - SAM2_start - 1) / (fps / 3)`, which 
    is then rounded and turned into an integer. 

    Parameters
    ----------
    annotations_file : str
        The full path to the annotation file
    fps : int
        The FPS of the unreduced video that the annotations were
        initially meant for
    SAM2_start : int
        Value the ensures the annotated frame value matches up 
        with the fames that will be ingested by SAM2
    df_columns : list of str
        Keys to extract from dictionary that will become the 
        DataFrame columns
    frame_col_name : str 
        The name in `df_columns` that corresponds to the 
        annotation frame column

    Returns
    -------
    Pandas.DataFrame
        DataFrame with columns `df_columns` with column 
        `frame_col_name` adjusted

    Examples
    --------
    >>> annotations_file = "./my_annotations.npy"
    >>> fps = 24
    >>> SAM2_start = 0
    >>> df_columns = ['Frame', 'ClickType', 'FishLabel', 'Location']
    >>> frame_col_name = 'Frame'
    >>> adjust_annotations(annotations_file, fps, SAM2_start, 
                           df_columns, frame_col_name)
    """

    # TODO: in formula 3 is hardcoded, we might want to change that

    # TODO: check that all inputs are correctly provided 

    # Read in npy file corresponding to dict of annotations
    annotations = np.load(annotations_file, allow_pickle=True)

    # Convert dict to DataFrame 
    df = pd.DataFrame(list(annotations))

    # Drop all columns, except those in df_columns 
    df = df[df_columns]

    # Correct annotation frame value, so it coincides with video frame value
    df[frame_col_name] = (df[frame_col_name] - SAM2_start - 1) / (fps / 3)

    # Round and convert frame value to an integer 
    # TODO: do we need to provide a warning that a rounding was necessary? 
    df[frame_col_name] = round(df[frame_col_name]).astype(int)

    return df  

def get_frame_chunks_df(df=None, obj_name=None, frame_name=None, click_type_name=None):
    """
    Using `click_type_name` column values of 3 and 4, obtains the enter and 
    exit frame values for each `obj_name`. Additionally, returns `df`
    with index `obj_name` and drops `click_type_name` rows with 
    values of 3 and 4. 

    Parameters
    ----------
    df : Pandas.DataFrame 
        The DataFrame representing the adjusted annotations 
        that will be chunked 
    obj_name : str
        A string representing the column of `df` that will 
        become the index of returned DataFrames and corresponds 
        to the object ID
    frame_name : str 
        The name that corresponds to the column that contains 
        frame values
    click_type_name : str
        The name of the column in `df` that corresponds to 
        the click type

    Returns
    -------
    obj_frame_chunks : Pandas.DataFrame
        DataFrame with index `obj_name` and columns 
        `EnterFrame` and `ExitFrame` representing the 
        frame the object enters and exits the scene, respectively 
    df : Pandas.DataFrame
        Input `df` with index `obj_name` and dropped `click_type_name` 
        rows with values of 3 and 4.

    Examples
    --------
    >>> obj_name = 'FishLabel'
    >>> frame_name = 'Frame'
    >>> click_type_name = 'ClickType'
    >>> get_frame_chunks_df(df, obj_name, frame_name, click_type_name)
    """

    # TODO: check types of inputs 

    # For each obj_name get frame where the object enters the scene 
    enter_frame = df[df[click_type_name] == 3][[obj_name, frame_name]].astype(int) 
    enter_frame = enter_frame.sort_values(by=[obj_name, frame_name], ascending=True)
    
    # For each obj_name get frame where the object exits the scene 
    exit_frame = df[df[click_type_name] == 4][[obj_name, frame_name]].astype(int) 
    exit_frame = exit_frame.sort_values(by=[obj_name, frame_name], ascending=True)

    # Check that each enter point has a corresponding exit point
    if (enter_frame.shape != exit_frame.shape) or (not np.array_equal(enter_frame[obj_name].values, exit_frame[obj_name].values)):
        raise RuntimeError(f"A {obj_name} does not have both an enter and exit point!")

    # Drop obj_name from exit_frame, now that we have sorted and compared them
    exit_frame.drop(columns=obj_name, axis=1, inplace=True)

    # Turn obj_name column back to a string 
    enter_frame[obj_name] = enter_frame[obj_name].astype(str) 

    # Concatenate columns to improve ease of use later
    obj_frame_chunks = pd.concat([enter_frame.reset_index(drop=True), exit_frame.reset_index(drop=True)], axis=1)
    obj_frame_chunks.columns = [obj_name, 'EnterFrame', 'ExitFrame']

    # Drop df rows that have click_type_name values of 3 or 4
    df = df[~df[click_type_name].isin([3, 4])]

    # Modify df so it has obj_name as its index
    df = df.set_index(obj_name)

    return obj_frame_chunks, df

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

def draw_and_save_frame_seg(bool_masks, jpg_save_dir, frame_paths, out_frame_idx, out_obj_ids, colors, 
                            font_size=75, font_color="red", alpha=0.6):
    """
    Draws segmentation masks on top of the frames and saves 
    the generated image to `img_save_dir`. 

    Parameters
    ----------
    bool_masks : Tensor of bools
        A tensor of shape (number of masks, frame pixel height, frame pixel width)
        representing the generated segmentation masks
    jpg_save_dir : str
        The path containing frames that will be overwritten with the masks
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

    # Get frame name using the stem of the frame JPG
    frame_id = Path(frame_paths[out_frame_idx]).stem

    # Draw each mask on top of image representing the frame
    image_w_seg = decode_image(jpg_save_dir + f"/{frame_id}.jpg")
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
    img_pil.save(jpg_save_dir + f"/{frame_id}.jpg")

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
