# TODO: Make it work when a character has no lines... or verify that it does work
# TODO: IMPORTANT: Ariel's "P" and "R" does not remove the correct line. 
#         Better idea is use the remove overlaying areas method? Maybe?
# TODO: Thinning is not applied to OpenCV methods

import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk, ImageDraw
from skeletonize import runFromGUI
import threading
import os
import subprocess
import platform
import shutil

skeletonize_methods = ["Default", "OpenCV 1", "OpenCV 2", "skimage skeletonization", "skimage thinning", "skimage medial-axis"]
font_path = None

# defaultCompression = 100
defaultCompression = 5 # We'll default to MUCH lower now that chars are stored in progmem
defaultErode = 6

def open_ino_file(file_path):
    file_path = os.path.abspath(file_path)
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    arduino_executable = None
    
    # Determine the OS and set the Arduino executable path
    if platform.system() == "Windows":
        arduino_executable = shutil.which("arduino.exe")
    elif platform.system() == "Darwin":  # macOS
        arduino_executable = shutil.which("arduino")
    elif platform.system() == "Linux":
        arduino_executable = shutil.which("arduino")

    if arduino_executable:
        try:
            subprocess.run([arduino_executable, file_path], check=True)
            return
        except subprocess.CalledProcessError:
            print("Failed to open file with Arduino IDE")

    # Fallback: Open with default program associated with .txt files
    try:
        if platform.system() == "Windows":
            os.startfile(file_path)
        elif platform.system() == "Darwin":
            subprocess.run(["open", file_path])
        else:
            subprocess.run(["xdg-open", file_path])
    except Exception as e:
        print(f"Failed to open file: {e}")

def submit_action():
    if not font_path:
        print("No font path specified")
        return
    
    loading_window = show_loading_overlay(root)
    root.update()

    try:
        exportedPath = runFromGUI(int(erode_amount.get()), skeletonization_options.get(), int(compression_strength.get()), font_path)
        print(exportedPath)

        open_ino_file(exportedPath)

    finally:
        # Close the loading overlay
        loading_window.destroy()

def show_loading_overlay(root):
    loading_window = tk.Toplevel(root)
    loading_window.geometry("400x150")
    loading_window.title("Loading")
    loading_window.transient(root)
    loading_window.grab_set()

    # Center the loading window on top of the root window
    x = root.winfo_x() + (root.winfo_width() // 2) - 200
    y = root.winfo_y() + (root.winfo_height() // 2) - 75
    loading_window.geometry(f"+{x}+{y}")

    tk.Label(loading_window, text="Processing...", font=("Helvetica", 16)).pack(pady=10)
    tk.Label(loading_window, text="Please allow up to a few minutes.", font=("Helvetica", 12)).pack()
    tk.Label(loading_window, text="If this is too long to wait, you can spend hours coding it :)", font=("Helvetica", 10)).pack(pady=(5, 0))

    # Make sure it stays on top
    loading_window.attributes("-topmost", True)
    
    return loading_window

def update_character(*args):
    char = char_input.get()
    # Keep the input as a single char
    if len(char) > 1:
        char = char[0]  # Keep only the first character
        char_input.delete(0, tk.END)  # Clear the input field
        char_input.insert(0, char)  # Insert the trimmed character back
    start_render_thread()

def update_erode_amount(*args):
    start_render_thread()

def update_compression_strength(value):
    start_render_thread()

def update_skeletonization_option(*args):
    start_render_thread()

# 
current_thread = None
def start_render_thread():
    global current_thread
    if current_thread and current_thread.is_alive():
        pass # Python apparently doesn't have a way to kill threads... smh 

    current_thread = threading.Thread(target=render_preview)
    current_thread.start()

def render_preview():
    if not font_path:
        return
    
    contoursJSON = runFromGUI(int(erode_amount.get()), skeletonization_options.get(), int(compression_strength.get()), font_path, True, char_input.get())
    contours = contoursJSON[list(contoursJSON.keys())[0]]

    outline_color = (0, 0, 0, 70)  # Black with 100/255 transparency
    outline_thickness = 15

    # Create a blank image with a white background
    image = Image.new("RGBA", (255, 255+(outline_thickness*2)), (255, 255, 255, 0)) # the char is 255 tall by default, plus the outline on both sides
    draw = ImageDraw.Draw(image, "RGBA")


    # Draw the semi-transparent outline
    def draw_line_with_caps(draw, start_point, end_point, color, thickness):
        # Draw the line
        draw.line([start_point, end_point], fill=color, width=thickness)
        
        # Draw circles at the start and end points to create rounded caps
        radius = thickness // 2
        draw.ellipse([start_point[0] - radius, start_point[1] - radius,
                      start_point[0] + radius, start_point[1] + radius], fill=color)
        draw.ellipse([end_point[0] - radius, end_point[1] - radius,
                      end_point[0] + radius, end_point[1] + radius], fill=color)

    # Draw the semi-transparent outline with rounded ends
    for contour in contours:
        for i in range(len(contour) - 1):
            # We also shift everything by outline_thickness so that the full outline is not cut off
            start_point = (int(contour[i][0]), int(contour[i][1]) + outline_thickness)
            end_point = (int(contour[i + 1][0]), int(contour[i + 1][1]) + outline_thickness)
            draw_line_with_caps(draw, start_point, end_point, outline_color, outline_thickness)
        
    # Draw the solid black lines on top
    solid_color = (0, 0, 0, 255)  # Solid black

    for contour in contours:
        for i in range(len(contour) - 1):
            # Again, shift down by outline thickness
            start_point = (int(contour[i][0]), int(contour[i][1]) + outline_thickness)
            end_point = (int(contour[i + 1][0]), int(contour[i + 1][1]) + outline_thickness)
            draw.line([start_point, end_point], fill=solid_color, width=0)
        
        # Optional: close the solid lines if needed
        # draw.line([start_point, (int(contour[0][0]), int(contour[0][1]))], fill=solid_color, width=1)

    # Convert to image and update the panel
    img_tk = ImageTk.PhotoImage(image)
    final_image_panel.config(image=img_tk)
    final_image_panel.image = img_tk  # Keep a reference to avoid garbage collection


def load_font_file():
    global font_path  # Add this line
    file_path = filedialog.askopenfilename(filetypes=[("Font Files", "*.ttf;*.otf"), ("TrueType Font Files", "*.ttf"), ("OpenType Font Files", "*.otf")])
    if file_path:
        font_path = file_path  # Save the path
        file_name = file_path.split("/")[-1]  # Extract filename only
        file_label.config(text=f"Selected File:\n{file_name}")
        # Trigger an initial update to display the character
        update_character()

        submit_button.config(state=tk.NORMAL, bg="green")


def create_gui():
    global root
    root = tk.Tk()
    root.title("AnyFont")
    root.geometry("1000x600")  # Adjust size as needed

    # Frame for controls
    control_frame = tk.Frame(root, padx=20, pady=20)
    control_frame.grid(row=0, column=0, sticky="nsew")

    # Erode Amount
    tk.Label(control_frame, text="Erode (thins char):").grid(row=0, column=0, sticky="w")
    global erode_amount
    erode_amount = tk.Spinbox(control_frame, from_=0, to=100, increment=1)
    erode_amount.grid(row=0, column=1, padx=10, pady=10)
    erode_amount.delete(0, tk.END)  # Clear the current text
    erode_amount.insert(0, str(defaultErode))  # Set default value
    erode_amount.bind("<KeyRelease>", lambda event: update_erode_amount())
    erode_amount.bind("<ButtonRelease-1>", lambda event: update_erode_amount())

    # Skeletonization Options
    tk.Label(control_frame, text="Skeletonization Method:").grid(row=1, column=0, sticky="w", pady=(10, 5))
    global skeletonization_options
    skeletonization_options = ttk.Combobox(control_frame, values=skeletonize_methods, state="readonly")
    skeletonization_options.set(skeletonize_methods[0])  # Default to the first option
    skeletonization_options.grid(row=1, column=1, padx=10, pady=5)
    skeletonization_options.bind("<<ComboboxSelected>>", update_skeletonization_option)

    # Compression Strength
    tk.Label(control_frame, text="Compression Strength:").grid(row=2, column=0, sticky="w", pady=10)
    global compression_strength
    compression_strength = tk.Scale(control_frame, from_=0, to=180, orient=tk.HORIZONTAL, length=360)
    compression_strength.set(defaultCompression)
    compression_strength.grid(row=2, column=1, padx=10, pady=10)

    compression_strength.bind("<ButtonRelease-1>", lambda event: update_compression_strength(compression_strength.get()))


    # File Input
    tk.Label(control_frame, text="Font File:").grid(row=3, column=0, sticky="w", pady=10)
    global file_label
    file_label = tk.Label(control_frame, text="No file selected")
    file_label.grid(row=3, column=1, padx=10, pady=5, sticky="w")
    tk.Button(control_frame, text="Browse", command=load_font_file).grid(row=3, column=2, padx=10, pady=5)

    global submit_button
    submit_button = tk.Button(control_frame, text="Submit", command=submit_action, height=2, width=15, bg="light grey", fg="white", font=("Helvetica", 16), state=tk.DISABLED)
    submit_button.grid(row=4, column=0, columnspan=3, pady=300, sticky="s")

    # Frame for rendering area
    render_frame = tk.Frame(root, padx=20, pady=20, bg="lightgray")
    render_frame.grid(row=0, column=1, sticky="nsew")

    # Single Character Input
    tk.Label(render_frame, text="Preview Character:").grid(row=0, column=0, sticky="w", pady=10)
    global char_input
    char_input = tk.Entry(render_frame, width=10)
    char_input.grid(row=0, column=1, padx=10, pady=10)
    char_input.insert(0, "A")
    # char_input
    char_input.bind("<KeyRelease>", update_character)

    # Final Image Panel
    global final_image_panel
    final_image_panel = tk.Label(render_frame, bg="white")
    final_image_panel.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

    # Configure grid weights for resizing
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=3)

    render_frame.grid_rowconfigure(1, weight=1)
    render_frame.grid_columnconfigure(0, weight=1)
    render_frame.grid_columnconfigure(1, weight=1)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
