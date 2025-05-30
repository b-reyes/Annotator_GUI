import time
import numpy as np
import cv2
from tkinter import *
from tkinter import ttk, filedialog
from tkinter import messagebox
import customtkinter as ctk
from PIL import Image, ImageTk
import csv
import os
import pandas as pd

# Initialize variables
current_frame_index = [0]
xLocation = [0]
yLocation = [0]
clickType = [1]  # Default to positive click (1)
fishLabel = [0]
paused = [False]
annotations = []
video_speed = 1.0  # Playback speed multiplier
frames = []
vid_height, vid_width = 0, 0
fps = 30  # Default FPS, will update dynamically based on video
special_frame_start = 0  # Default starting frame for SAM2
special_frame_interval = 10  # Default, will calculate dynamically
fish_family = ["Parrotfish"]  # Default fish family

#Define video player size. Should be x = y * 1.5
video_size_x=600
video_size_y=400

# Create the main window
root = ctk.CTk()
root.title("Video Annotation GUI")
root.geometry("1200x700") #May need to change this line to fit different computer screens

# Play and Pause Function
def pause():
    paused[0] = not paused[0]
    button_play_pause.configure(text="Pause ||" if not paused[0] else "Play ▶")
    if not paused[0]:
        play_video()

# Load Video Function
def load_video():
    global frames, vid_height, vid_width, fps, special_frame_interval, video_size_x, video_size_y

    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi")])
    if not file_path:
        return

    cap = cv2.VideoCapture(file_path)
    frames.clear()

    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Update FPS dynamically
    special_frame_interval = max(1, round(fps) / 3)  # Calculate interval for 3 frames per second

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (video_size_x, video_size_y))
            frames.append(frame)
        else:
            break

    vid_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    vid_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap.release()

    slider_frame.configure(to=len(frames) - 1)
    play_video()

# Canvas Click Event
def canvas_click_events(event):
    x_coord = (vid_width / video_size_x) * event.x
    y_coord = (vid_height / video_size_y) * event.y
    xLocation[0] = x_coord
    yLocation[0] = y_coord

# Add Annotation Function
def add_annotation():
    fishLabel[0] = fish_name.get()
    annotation = {
        "Frame": current_frame_index[0],
        "clickType": clickType[0],
        "fishLabel": fishLabel[0],
        "Fish_Fam": fish_family[0],
        "Location": np.array([round(xLocation[0], 3), round(yLocation[0], 3)])
    }
    annotations.append(annotation)
    update_annotation_table()

    print(f"Annotation added for {current_frame_index[0]}.")

# Update Annotations Table
def update_annotation_table():
    for row in treeview.get_children():
        treeview.delete(row)
    for i, annotation in enumerate(annotations):
        treeview.insert(
            "",
            "end",
            iid=i,
            values=(
                i,
                annotation["Frame"],
                annotation["clickType"],
                annotation["fishLabel"],
                annotation["Fish_Fam"],
                annotation["Location"][:2],
            ),
        )
# Delete Selected Annotations
def delete_selected():
    selected_items = treeview.selection()
    for item in sorted(selected_items, reverse=True):
        annotations.pop(int(item))
    update_annotation_table()

# Delete All Annotations
def delete_all():
    response = messagebox.askquestion(
        "Confirm Delete All",
        "Are you sure you want to delete all annotations?",
        icon="warning"
    )
    if response == "yes":
        annotations.clear()
        update_annotation_table()

# Toggle Click Type
def toggle_click_type():
    if clickType[0] == 1:
        clickType[0] = 0
        button_toggle_click.configure(text="Negative Click")
    elif clickType[0] == 0:
        clickType[0] = 2
        button_toggle_click.configure(text="Bite")
    else:
        clickType[0] = 1
        button_toggle_click.configure(text="Positive Click")

# Toggle Fish Family
def toggle_fish_family():
    if fish_family[0] == "Parrotfish":
        fish_family[0] = "Surgeonfish"
        button_toggle_fish_family.configure(text="Surgeonfish")
    elif fish_family[0] == "Surgeonfish":
        fish_family[0] = "Damselfish"
        button_toggle_fish_family.configure(text="Damselfish")
    elif fish_family[0] == "Damselfish":
        fish_family[0] = "Other"
        button_toggle_fish_family.configure(text="Other")
    else:
        fish_family[0] = "Parrotfish"
        button_toggle_fish_family.configure(text="Parrotfish")


# Import Previous Annotations Function
def import_annotations():

    # Open a file dialog to select a .npy file
    file_path = filedialog.askopenfilename(
        title = "Import Previous Annotations",
        filetypes=(("Numpy Files", "*.npy"), ("CSV Files", "*.csv"), ("All Files", "*.*"))
    )
#Check if the user selected a file or canceled the dialog
    if not file_path:
        return

    try:
        file_extension = os.path.splitext(file_path)[1].lower()

        #Case 1: Load .npy file
        if file_extension == ".npy":
            imported_annotations=np.load(file_path, allow_pickle=True)

            #Check if the loaded file has the expected format
            if not isinstance(imported_annotations, np.ndarray):
                raise ValueError("The selected file does not contain compatible annotation data")
        
        #Append annotations to existing list
            for annotation in imported_annotations:
                if isinstance(annotation, dict) and all(key in annotation for key in ["Frame", "clickType", "fishLabel", "Fish_Fam", "Location"]):
                    annotations.append(annotation)
                else:
                    raise ValueError("One or more annotations in the file have an invalid format.")
        
        #Case 2: Load .csv file
        elif file_extension == ".csv":
            imported_annotations = pd.read_csv(file_path)

            #check if the necessary columns are in the .csv
            required_columns = ["Frame", "clickType", "fishLabel", "Fish_Fam", "Location"]
            if not all(col in imported_annotations.columns for col in required_columns):
                raise ValueError(f"The CSV file must contain the following columns: {', '.join(required_columns)}.")
            for _, row in imported_annotations.iterrows():
                location_str = row["Location"]
                location = eval(location_str) if isinstance(location_str, str) else location_str
                annotation = {
                    "Frame": int(row["Frame"]),
                    "clickType": row["clickType"],
                    "fishLabel": row["fishLabel"],
                    "Fish_Fam": row["Fish_Fam"],
                    "Location": np.array(location)
                }
                annotations.append(annotation)
                        #Update the annotation table with new data

        else:
            raise ValueError("The selected file is neither a valid .npy nor .csv file.")

        #Update the annotation table with new data
        update_annotation_table()
        # Optionally, show a message to the user that the import was successful
        messagebox.showinfo("Import Successful", f"Successfully imported {len(imported_annotations)} annotations.")

    except Exception as e:
        # Handle any errors (e.g., file not found, invalid format, etc.)
        messagebox.showerror("Error", f"An error occurred while importing annotations: {str(e)}")

# Save Annotations Function
def save_annotations():
    file_name = file_name_var.get().strip() or "annotations"  # Default name if none provided
    general_annotations = [
        a for a in annotations if a["clickType"] in [0, 1]
    ]
    bite_annotations = [
        a for a in annotations if a["clickType"] == 2
    ]

    # Save general annotations to "<file_name>_annotations.npy"
    np.save(f"{file_name}_annotations.npy", general_annotations)

    # Save bites to "<file_name>_bites.csv"
    with open(f"{file_name}_bites.csv", "w", newline="") as csvfile:
        fieldnames = ["Frame", "clickType", "fishLabel", "Fish_Fam", "Location"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for annotation in bite_annotations:
            writer.writerow({
                "Frame": annotation["Frame"],
                "clickType": annotation["clickType"],
                "fishLabel": annotation["fishLabel"],
                "Fish_Fam": annotation["Fish_Fam"],
                "Location": annotation["Location"].tolist()
            })
    messagebox.showinfo("Save Successful.", f"{len(general_annotations)} location annotations saved as '{file_name}_annotations.npy and {len(bite_annotations)} bites saved as '{file_name}_bites.csv'.")

# Play Video
playing_task = None

def play_video():
    global playing_task

    if paused[0] or current_frame_index[0] >= len(frames):
        return

    frame_rgb = cv2.cvtColor(frames[current_frame_index[0]], cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    photo = ImageTk.PhotoImage(image)

    label_video.configure(image=photo)
    label_video.image = photo

    slider_frame.set(current_frame_index[0])
    update_time_display()

    current_frame_index[0] += 1
    playing_task = root.after(int(1000 / (fps * video_speed)), play_video)

# Update Frame from Slider
def update_frame_from_slider(event):
    global playing_task
    if playing_task is not None:
        root.after_cancel(playing_task)
    current_frame_index[0] = int(slider_frame.get())
    frame_rgb = cv2.cvtColor(frames[current_frame_index[0]], cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    photo = ImageTk.PhotoImage(image)
    label_video.configure(image=photo)
    label_video.image = photo
    update_time_display()

# Update Time Display
def update_time_display():
    frame_num = current_frame_index[0]
    time_in_seconds = frame_num / fps
    time_display_var.set(f"Time: {time_in_seconds:.2f}s | Frame: {frame_num} | Speed: {video_speed:.1f}x")

    if frame_num >= special_frame_start and (frame_num - special_frame_start) % special_frame_interval == 0:
        special_frame_var.set("SAM2 Frame: Annotate Fish Position")
        label_special_frame.configure(font=("Arial", 14, "bold"), fg="red")
    else:
        special_frame_var.set("----")
        label_special_frame.configure(font=("Arial", 12), fg="black")

# Advance Frame

def advance_frame(delta):
    global playing_task
    if playing_task is not None:
        root.after_cancel(playing_task)
    current_frame_index[0] = max(0, min(len(frames) - 1, current_frame_index[0] + delta))
    frame_rgb = cv2.cvtColor(frames[current_frame_index[0]], cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    photo = ImageTk.PhotoImage(image)
    label_video.configure(image=photo)
    label_video.image = photo
    update_time_display()

# Navigate to Next Special Frame
def next_special_frame():
    global playing_task
    if playing_task is not None:
        root.after_cancel(playing_task)
    while current_frame_index[0] < len(frames):
        current_frame_index[0] += 1
        if (current_frame_index[0] - special_frame_start) % special_frame_interval == 0:
            break
    frame_rgb = cv2.cvtColor(frames[current_frame_index[0]], cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    photo = ImageTk.PhotoImage(image)
    label_video.configure(image=photo)
    label_video.image = photo
    update_time_display()

#Navigate to Previous Special Frame
def prev_special_frame():
    global playing_task
    if playing_task is not None:
        root.after_cancel(playing_task)
    while current_frame_index[0] > 0:
        current_frame_index[0] -= 1
        if (current_frame_index[0] - special_frame_start) % special_frame_interval == 0:
            break
    frame_rgb = cv2.cvtColor(frames[current_frame_index[0]], cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    photo = ImageTk.PhotoImage(image)
    label_video.configure(image=photo)
    label_video.image = photo
    update_time_display()
    
# Adjust Playback Speed
def adjust_speed(delta):
    global video_speed
    video_speed = max(0.1, video_speed + delta)
    update_time_display()

def reset_speed():
    global video_speed
    video_speed = 1.0
    update_time_display()

# UI Layout
root.bind("<Return>", lambda event: add_annotation())
root.bind("<Left>", lambda event: prev_special_frame())
root.bind("<Right>", lambda event: next_special_frame())

frame_controls = ctk.CTkFrame(root)
frame_controls.pack(side=LEFT, fill=Y, padx=10, pady=10)

button_browse = ctk.CTkButton(frame_controls, text="Browse Video", command=load_video)
button_browse.pack(pady=10)

Label(frame_controls, text="SAM2 Start Frame:").pack(pady=5)
special_frame_start_var = IntVar(value=0)
special_frame_entry = ttk.Entry(frame_controls, textvariable=special_frame_start_var)
special_frame_entry.pack(pady=5)

def update_special_frame_start():
    global special_frame_start
    special_frame_start = special_frame_start_var.get()

button_set_special_frame = ctk.CTkButton(frame_controls, text="Set SAM2 Frame", command=update_special_frame_start)
button_set_special_frame.pack(pady=10)

button_toggle_click = ctk.CTkButton(frame_controls, text="Positive Click", command=toggle_click_type)
button_toggle_click.pack(pady=10)

button_toggle_fish_family = ctk.CTkButton(
    frame_controls,
    text="Parrotfish",
    command=toggle_fish_family
)
button_toggle_fish_family.pack(pady=10)
Label(frame_controls, text="Fish Name:").pack(pady=5)
fish_name = StringVar()
entry_fish_name = ttk.Entry(frame_controls, textvariable=fish_name)
entry_fish_name.pack(pady=5)


button_add_annotation = ctk.CTkButton(frame_controls, text="Add Annotation", command=add_annotation)
button_add_annotation.pack(pady=10)

button_delete_selected = ctk.CTkButton(frame_controls, text="Delete Selected", command=delete_selected)
button_delete_selected.pack(pady=10)

button_delete_all = ctk.CTkButton(frame_controls, text="Delete All", command=delete_all)
button_delete_all.pack(pady=10)

button_import = ctk.CTkButton(frame_controls, text="Import Previous Annotations", command=import_annotations)
button_import.pack(pady=10)

Label(frame_controls, text="Saving File Name:").pack(pady=5)
file_name_var = StringVar()
entry_file_name = ttk.Entry(frame_controls, textvariable=file_name_var)
entry_file_name.pack(pady=5)



button_save_annotations = ctk.CTkButton(frame_controls, text="Save Annotations", command=save_annotations)
button_save_annotations.pack(pady=10)

# Central Playback and Info Controls
frame_central_controls = ctk.CTkFrame(root)
frame_central_controls.pack(side=TOP, fill=X, padx=10, pady=10)

button_play_pause = ctk.CTkButton(frame_central_controls, text="Pause ||", command=pause)
button_play_pause.pack(pady=5, side=LEFT)

frame_bottom_controls = ctk.CTkFrame(root)
frame_bottom_controls.pack(side=TOP, fill=X)

button_prev_special_frame = ctk.CTkButton(frame_bottom_controls, text="Prev SAM2 Frame", command=prev_special_frame)
button_prev_special_frame.pack(side=LEFT, padx=5, pady=5)

button_prev_frame = ctk.CTkButton(frame_central_controls, text="<< Prev Frame", command=lambda: advance_frame(-1))
button_prev_frame.pack(pady=5, side=LEFT)

button_next_frame = ctk.CTkButton(frame_central_controls, text="Next Frame >>", command=lambda: advance_frame(1))
button_next_frame.pack(pady=5, side=LEFT)



button_next_special_frame = ctk.CTkButton(frame_bottom_controls, text="Next SAM2 Frame", command=next_special_frame)
button_next_special_frame.pack(side=LEFT, padx=5, pady=5)

button_decrease_speed = ctk.CTkButton(frame_central_controls, text="- Speed", command=lambda: adjust_speed(-0.1))
button_decrease_speed.pack(pady=5, side=LEFT)

button_increase_speed = ctk.CTkButton(frame_central_controls, text="+ Speed", command=lambda: adjust_speed(0.1))
button_increase_speed.pack(pady=5, side=LEFT)

button_reset_speed = ctk.CTkButton(frame_central_controls, text="Reset Speed", command=reset_speed)
button_reset_speed.pack(pady=5, side=LEFT)

frame_video = ctk.CTkFrame(root)
frame_video.pack(side=LEFT, fill=BOTH, expand=True, padx=10, pady=10)

label_video = ctk.CTkLabel(frame_video, text="")  # Remove CTkLabel watermark
label_video.pack(padx=10, pady=10)
label_video.bind('<Button-1>', canvas_click_events)

slider_frame = ttk.Scale(frame_video, from_=0, to=0, orient=HORIZONTAL, length=video_size_x)
slider_frame.pack(pady=10)
slider_frame.bind("<ButtonRelease-1>", update_frame_from_slider)

#Display Below Video Player
frame_info_display = ctk.CTkFrame(root)
frame_info_display.pack(side=TOP, fill=X, padx=10, pady=10, after=frame_video)

special_frame_var = StringVar()
special_frame_var.set("----")
label_special_frame = Label(frame_info_display, textvariable=special_frame_var, font=("Arial", 12))
label_special_frame.pack(pady=5)

time_display_var = StringVar()
time_display_var.set("Time: 0.00s | Frame: 0 | Speed: 1.0x")
label_time_display = Label(frame_info_display, textvariable=time_display_var, font=("Arial", 12))
label_time_display.pack(pady=5)

frame_annotations = ctk.CTkFrame(root)
frame_annotations.pack(side=RIGHT, fill=Y, padx=10, pady=10)

columns = ("ID", "Frame", "Click Type", "Fish Label", "Fish_Fam", "Coordinates")
treeview = ttk.Treeview(frame_annotations, columns=columns, show="headings")
for col in columns:
    treeview.heading(col, text=col)
    treeview.column(col, width=150)  # Adjust column width for readability
treeview.pack(fill=BOTH, expand=True)

root.mainloop()
