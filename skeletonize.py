from PIL import Image, ImageDraw, ImageFont
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import cv2
import numpy as np
import CodeBuilder

# I got annoyed
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, message=r'.+binary_hit_or_miss.+')

# Various skeletonizers 
from scipy.ndimage import binary_erosion
from skimage.morphology import medial_axis, skeletonize, thin
import cv2.ximgproc
import scipy.ndimage.morphology as m
from scipy.ndimage import generate_binary_structure



"""
Process:

I loop through the alphabet,
For each letter, render it on a blank background.
Erode the character in a given amount of times so that the lines are thinner and easier to trace.
From there, we use one of a variety of "skeletonization" methods to extract just the center line of the character.

This line is actually a an array of letter segments
And sometimes these lines cover the same area twice
So first we remove the smaller duplicate lines
Each of these letter segments forms a closed loop - in the character "X", each arm is traced up and then down, back to where it started
Then we need to compress it somewhat to fit in the arduino.

In order for some letters to look right, the width has to match the pen width... 
 so might need to scale down how large it prints and buffer the outside.
 Ideally the label maker code will just write on a 255x scale, with width working proportionally 

"""

# You can modify these
erode_iterations = 8                # "erodeIterations" essentially makes the font thinner so it's easier to trace - but too low, and it some parts of the character will disappear
erode_sensitive_letters = ["B"]     # Letters that look bad when eroded too much
preview_erode_and_exit = False      # Preview the erosion of the first first alphabet character for testing numbers
font_path = 'comic.ttf'             # Your font path
show_letters = ["A"]                # Letters to show what they look like when they are processed
show_all = False                    # Ignore the former array and just display all processed chars
line_combine_degree_threshold = 120 # This sets how many degrees off one line can be from the next to be compressed into one. Higher = more compressed. Degrees off accumulate, so it should still be high quality with high numbers
skeletonize_method_index = 0        # See `skeletonize_methods` below

# Only change things beyond here if you know what you are doing 
overlap_remove_threshold = 0.9              # Sometimes multiple contours are produced for the same area, we'll remove ones that overlap a ton
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"     # If you don't need characters here you can remove them to save ram - it currently does not handle lowercase very well on a variety of levels 
# alphabet = "A"
letter_buffer = 10                          # buffer around of letter 
min_uncompressed_segment_size = 4 if line_combine_degree_threshold <= 10 else 0     # Prefilter uncompressed segments smaller than a specified size
min_compressed_segment_size   = min_uncompressed_segment_size                       # Postfilter compressed segments smaller than a specified size
filter_below_len = 50
outfile = "AnyFont.ino"
ignore_height = []                  # Some letters like Q have a single part go lower than the rest... but this causes issues drawing it so we won't use it for now
skeletonize_methods = ["Default", "OpenCV 1", "OpenCV 2", "skimage skeletonization", "skimage thinning", "skimage medial-axis"]

# Don't touch anything beyond here
font = ImageFont.truetype(font_path, 400)

# Utility methods
def flatten_list(nested_list):
    """Flatten a nested list."""
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))  # Recursively flatten
        else:
            flat_list.append(item)
    return flat_list

def render(image):
    # Drop in render function for debugging
    if isinstance(image, Image.Image): 
        image = np.array(image)
    elif isinstance(image, cv2.UMat): 
        image = cv2.UMat.get(image)

    # Create a border by extending the extent of the image
    border_size = 5  # Border size in pixels
    height, width = image.shape[:2]
    extent = [-border_size, width + border_size, -border_size, height + border_size]
    
    plt.imshow(image, cmap='gray', extent=extent)
    plt.gca().add_patch(plt.Rectangle((0, 0), width, height, linewidth=1, edgecolor='black', facecolor='none'))
    plt.axis('off')  # Hide the axes
    plt.show()
    exit()

def cv2SkeletonizeMethods(image):
    numpy_image = np.array(image)
    skeleton = cv2.UMat(numpy_image)
    skeleton = cv2.bitwise_not(skeleton) # CV2 does skeletonization backwards
    # Possible cv2 methods will be here, since we need to convert to a cv2 array then back to use them a drop-in replacements
    # 

    if skeletonize_method_index == 1: skeleton = cv2.ximgproc.thinning(skeleton, thinningType=cv2.ximgproc.THINNING_GUOHALL)

    if skeletonize_method_index == 2: skeleton = cv2.ximgproc.thinning(skeleton, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    # 
    if (not type(skeleton) is np.ndarray): skeleton = cv2.UMat.get(skeleton)
    pil_image = Image.fromarray(skeleton)
    return pil_image

def customSkeletonize1(img):
    # Pulled from https://stackoverflow.com/questions/30026287/skeletonizing-an-image-in-python#30026551
    h1 = np.array([[0, 0, 0],[0, 1, 0],[1, 1, 1]]) 
    m1 = np.array([[1, 1, 1],[0, 0, 0],[0, 0, 0]]) 
    h2 = np.array([[0, 0, 0],[1, 1, 0],[0, 1, 0]]) 
    m2 = np.array([[0, 1, 1],[0, 0, 1],[0, 0, 0]])    
    hit_list = [] 
    miss_list = []
    for k in range(4): 
        hit_list.append(np.rot90(h1, k))
        hit_list.append(np.rot90(h2, k))
        miss_list.append(np.rot90(m1, k))
        miss_list.append(np.rot90(m2, k))    
    img = img.copy()
    while True:
        last = img
        for hit, miss in zip(hit_list, miss_list): 
            hm = m.binary_hit_or_miss(img, hit, miss) 
            img = np.logical_and(img, np.logical_not(hm)) 
        if np.all(img == last):  
            break

    if (not type(img) is np.ndarray): img = cv2.UMat.get(img)
    pil_image = Image.fromarray(img)
    return pil_image

def contour_area(contour):
    return cv2.contourArea(contour)

def calculate_overlap(box1, box2):
    # Find the intersection area
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[0] + box1[2], box2[0] + box2[2])
    y_bottom = min(box1[1] + box1[3], box2[1] + box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No overlap

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]

    smallest_area = min(box1_area, box2_area)

    return intersection_area / smallest_area

def remove_overlapping_contours(lines, overlap_threshold=0.9):
    bounding_boxes = [cv2.boundingRect(contour) for contour in lines]
    
    keep = [True] * len(lines)
    
    for i in range(len(lines)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(lines)):
            if not keep[j]:
                continue
            overlap_ratio = calculate_overlap(bounding_boxes[i], bounding_boxes[j])
            if overlap_ratio > overlap_threshold:
                # Remove the smaller contour
                if contour_area(lines[i]) > contour_area(lines[j]):
                    keep[j] = False
                else:
                    keep[i] = False
    
    # Filter out contours marked for removal
    filtered_lines = [lines[i] for i in range(len(lines)) if keep[i]]
    
    return filtered_lines

def calculate_direction(points):
    """Calculate the direction vector of a group of points."""
    start = points[0]
    end = points[-1]
    direction = end - start
    return direction / np.linalg.norm(direction)  # Normalize to get the direction vector

def calculate_distance(points):
    """
    Calculate the total Euclidean distance of a contour given its points.
    """
    distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    return np.sum(distances)

def compressLetter(letterSegments, group_size=3, threshold=5):
    # The basic idea here is to combine line segments that are within a threshold of degrees off
    # Degrees off accumulate each time we combine a line, preserving some slight curve in longer lines.

    compressedLetterSegments = []

    for contour in letterSegments:
        if len(contour) < group_size + 1:
            compressedLetterSegments.append(contour)  # Not enough points to compress

        compressed = [contour[0]]  # Start with the first point
        accumulated_offset = 0

        i = 0
        while i < len(contour) - group_size:
            current_group = contour[i:i + group_size]
            next_group = contour[i + 1:i + 1 + group_size]

            if len(next_group) < group_size:
                break  # Not enough points left for another full group

            dir1 = calculate_direction(current_group)
            dir2 = calculate_direction(next_group)

            angle_diff = np.degrees(np.arccos(np.clip(np.dot(dir1, dir2), -1.0, 1.0)))

            if angle_diff + accumulated_offset < threshold:
                accumulated_offset += angle_diff
            else:
                compressed.append(contour[i + group_size // 2])
                accumulated_offset = 0  # Reset when not combining

            i += 1

        # Append the last point in the contour
        compressed.append(contour[-1])
        compressedLetterSegments.append(np.array(compressed))
    
    return compressedLetterSegments

def split_contours_on_revisit(contours):
    visited_points = set()
    new_contours = []
    current_contour = []

    lastVisitedPoint = None

    for contour in contours:
        for point in contour:
            point_tuple = tuple(point)

            if point_tuple in visited_points:
                # If we just hit a new point after tracing a contour, save and empty current contour
                if current_contour and current_contour:
                    new_contours.append(np.array(current_contour, dtype=np.int32))
                current_contour = []
            else:
                # If we only just hit a new point, start this new contour on the last visited point
                if not current_contour and lastVisitedPoint is not None:
                    current_contour.append(lastVisitedPoint.tolist())

                # Add this new point to the contour
                current_contour.append(point.tolist())

                # Mark this new point as visited
                visited_points.add(point_tuple)

            lastVisitedPoint = point
        
        # Add remaining contour we were just working on, if there is one
        if current_contour:
            new_contours.append(np.array(current_contour, dtype=np.int32))
            current_contour = []

    return new_contours

def adjust_scale_and_center(lettersJson, previewing=False, x_buffer=0):
    # Written by me but it was tedious so I gave it to ChatGPT to make it add scaling letters lol
    # This whole thing could be MAJORLY more efficient but hey it works

    # K so the label maker uses x,y backwards so we need to flip x around here so it gets scaled and all properly
    # if not previewing:
    #     for letterStr in lettersJson:
    #         letterLines = lettersJson[letterStr]
    #         # Grab max x
    #         maxX = 0
    #         for contour in letterLines:
    #             for point in contour:
    #                 maxX = max(maxX, point[0])
    #         # Use max x to exactly invert - this could be more efficient but is most logically straightforward
    #         for contour in letterLines:
    #             for point in contour:
    #                 point[0] = (maxX - point[0])


    # Step 1: Flatten all points across all letters to find global min/max
    all_points = []
    for letter, contours in lettersJson.items():
        for contour in contours:
            all_points.extend(contour)
    
    all_points = np.array(all_points)
    
    # Find global min and max values
    min_x = np.min(all_points[:, 0])
    min_y = np.min(all_points[:, 1])

    max_value = np.max(all_points)
    
    # Step 2: Adjust each letter's contours
    # adjusted_letters = {}
    # for letter, contours in lettersJson.items():
    #     adjusted_contours = []
    #     for contour in contours:
    #         contour = np.array(contour)
    #         # Adjust x and y
    #         contour[:, 0] -= (min_x - x_buffer)
    #         contour[:, 1] -= min_y
    #         adjusted_contours.append(contour.tolist())
    #     adjusted_letters[letter] = adjusted_contours

    # Pull to 0,0 ish
    for letterStr in lettersJson:
        letterLines = lettersJson[letterStr]
        for contour in letterLines:
            for point in contour:
                point[0] -= (min_x - x_buffer)
                point[1] -= min_y

    # TODO maybe pull ignore_height letters halfway higher here? And slice off the rest, or compress it, or just use as is?
    
    # Step 3: Find new global max value after adjustments
    new_all_points = []
    for letter, contours in lettersJson.items():
        for contour in contours:
            new_all_points.extend(contour)
    
    new_all_points = np.array(new_all_points)
    new_max_value = np.max(new_all_points)
    
    # Scale all points uniformly to fit within 0 to 255
    scale_factor = 255 / new_max_value
    scaled_letters = {}
    for letter, contours in lettersJson.items():
        scaled_contours = [] # np.array()?
        for contour in contours:
            contour = np.array(contour)
            contour = np.floor(contour * scale_factor).astype(int)
            scaled_contours.append(contour.tolist())
        
        scaled_letters[letter] = scaled_contours
    

    # Find point closest to y=255, divide that by half, move all chars that half closer in that direction
    # TODO: figure out if fonts with smaller letteres need to be floored.... ideally not if we did the above right.
    # We don't need the following since we're now going to scale X to be the max no matter what... although this could be a scale-down option
    # largestY = 0
    # for letter, contours in scaled_letters.items():
    #     for contour in contours:
    #         y_values = np.array(contour)[:, 1]
    #         y_max = np.max(y_values)
    #         if y_max > largestY: largestY = y_max
    ## Buffer top to center letters (without sucking small ones to the top)
    # bottomGap = 255 - largestY
    # yTopBuffer = bottomGap // 2
    # for letterStr in scaled_letters:
    #     letterLines = scaled_letters[letterStr]
    #     for contour in letterLines:
    #         for point in contour:
    #             point[1] += yTopBuffer

    print("Done scaling")
    return scaled_letters

    # ChatGPT sucks at coding
    # Step 4: Find the tallest character
    # tallest_char_height = 0
    # tallest_char_y_min = float('inf')
    # tallest_char_y_max = float('-inf')
    
    # for letter, contours in scaled_letters.items():
    #     for contour in contours:
    #         y_values = np.array(contour)[:, 1]
    #         y_min = np.min(y_values)
    #         y_max = np.max(y_values)
    #         height = y_max - y_min
    #         if height > tallest_char_height:
    #             tallest_char_height = height
    #             tallest_char_y_min = y_min
    #             tallest_char_y_max = y_max
    
    # Step 5: Center the tallest character
    # y_center_target = (255 - tallest_char_height) / 2
    # y_shift = y_center_target - tallest_char_y_min
    
    # # Step 6: Apply vertical centering to all characters
    # centered_letters = {}
    # for letter, contours in scaled_letters.items():
    #     centered_contours = []
    #     for contour in contours:
    #         contour = np.array(contour)
    #         contour[:, 1] = contour[:, 1] + y_shift + y_padding
    #         centered_contours.append(contour.tolist())
        
    #     centered_letters[letter] = centered_contours
    

# Display skeletons for debugging
def plotCharacter(input):
    # Allow input of array or characters to render
    lines = None
    if isinstance(input, str): 
        lines = getRawLinesForCharacter(input)
    elif isinstance(input, np.ndarray):  
        lines = [input]
    else: 
        lines = input

    radius = 8
    plt.figure(figsize=(10, 5))

    colors = ['k', 'r', 'g', 'b', 'm']  # Example color list
    color_idx = 1

    outline = True
    if outline:
        for line in lines:
            x, y = [], []
            segments = []

            for point in line:
                if np.array_equal(point, [0, 0]):
                    # Add current segment to the outline collection
                    if len(x) > 1:
                        segments.append(np.column_stack((x, y)))
                    x, y = [], []  # Reset for the next segment
                else:
                    x.append(point[0])
                    y.append(point[1])

            # Add any remaining segment to the outline collection
            if len(x) > 1:
                segments.append(np.column_stack((x, y)))

            # Plot the outline with the custom radius
            if segments:
                outline = LineCollection(segments, linewidths=radius*2, colors='gray', alpha=0.5)
                plt.gca().add_collection(outline)

    # Now plot the lines with color changes at skips
    for line in lines:
        color_idx += 1
        current_color = colors[color_idx % len(colors)]
        x, y = [], []

        for point in line:
            if np.array_equal(point, [0, 0]):
                if len(x) > 1:
                    plt.plot(x, y, color=current_color, linewidth=2)
                x, y = [], []  # Reset for next segment
                color_idx += 1  # Change color after each skip
                current_color = colors[color_idx % len(colors)]
            else:
                x.append(point[0])
                y.append(point[1])

        # Plot any remaining segment
        if len(x) > 1:
            plt.plot(x, y, color=current_color, linewidth=2)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()
    plt.show()

# Methods to extract contours from characters
def getRawLinesForCharacter(char):
    # Draw the character over a white background
    image = Image.new("L", (600, 800), 255)
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), char, font=font, fill=0)

    # render(image)

    # Convert to binary for some skeletonizers
    binary_image = np.array(image) < 128

    # Thin down outlines to give skeletonization less room to play
    local_erode_iterations = erode_iterations
    struct = generate_binary_structure(2, 2)

    if char in erode_sensitive_letters: 
        local_erode_iterations /= 1.5
    
    if local_erode_iterations > 0: binary_image = binary_erosion(binary_image, structure=struct, iterations=int(local_erode_iterations)).astype(binary_image.dtype)
    if (preview_erode_and_exit): render(binary_image)

    # Skeletonize image

    # This is the smoothest and cleanest but it loses some detail
    if skeletonize_method_index == 1 or skeletonize_method_index == 2: skeleton = cv2SkeletonizeMethods(image)
    
    if skeletonize_method_index == 4: skeleton = thin(binary_image)
    if skeletonize_method_index == 3: skeleton = skeletonize(binary_image)

    # This one preforms well
    if skeletonize_method_index == 5: skeleton = medial_axis(binary_image)

    # Shows most character
    if skeletonize_method_index == 0: skeleton = customSkeletonize1(binary_image)

    # Convert skeleton to lines using OpenCV
    skeleton = img_as_ubyte(skeleton)
    contours, _ = cv2.findContours(skeleton, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Extract lines from OpenCV
    lines = [contour[:, 0, :] for contour in contours]
    return lines

def getProcessedLinesForInputChars(charsStr, previewing=False):
    # Build alphabet into lettersJson
    # charsStr = "A"
    lettersJson = {}
    totalRam = 0
    for letter in charsStr:
        print("=== Processing '" + letter + "' ===")

        # Extract lines
        letterSegments = getRawLinesForCharacter(letter)

        # Remove contours that are totally covered by other contours (like the circle inside "A" if the outside of the A already has the circle)
        letterSegments = remove_overlapping_contours(letterSegments, overlap_threshold=overlap_remove_threshold)

        # Record uncompressed size
        uncompressedSize = 0
        for segment in letterSegments:
            uncompressedSize += len(segment)

        # Remove portions that draw over already drawn areas (like when doing an arm or a leg of a letter)
        letterSegments = split_contours_on_revisit(letterSegments)
        letterSegments = [contour for contour in letterSegments if len(contour) >= min_uncompressed_segment_size] # The splitting method can be messy at times

        # Compress lines (turn tiny zig-zagging segments and lines heading in the same general direction into a single line)
        letterSegments = compressLetter(letterSegments, 3, line_combine_degree_threshold)
        letterSegments = [contour for contour in letterSegments if len(contour) >= min_compressed_segment_size]
        
        # Run another filter method for higher compressed fonts (higher compression means smaller segment sizes, but they span a greater area)
        letterSegments = [contour for contour in letterSegments if calculate_distance(contour) >= filter_below_len]

        # Info
        totalPoints = 0
        for segment in letterSegments:
            totalPoints += len(segment)
        
        ramUsed = totalPoints*2
        totalRam += ramUsed
        print(letter + ": " + str(uncompressedSize)    + "\tpoints (uncompressed)")
        print(letter + ": " + str(totalPoints)         + "\tpoints (compressed)")
        print(letter + ": " + str(ramUsed)       + "\tbytes  (roughly?)") # each x,y point uses 2 bytes
        print(letter + ": " + str(len(letterSegments)) + "\tsegments")
        
        # Process points into C code
        lettersJson[letter] = letterSegments

    # Rescale letters to be within 
    print("=== Rescaling letters ===")
    lettersJson = adjust_scale_and_center(lettersJson, previewing)

    # plotCharacter(lettersJson['E'])
    # plotCharacter(lettersJson['Q'])

    print("=== FINISHED ===")
    print("Font takes " + str(totalRam) + " bytes")

    return lettersJson


# Go through extracting for each char, as well as touching up the lines
def getLinesForAlphabetAndExport():
    lettersJson = getProcessedLinesForInputChars(alphabet)
    
    code = CodeBuilder.buildCode(lettersJson, ignore_height)

    with open(outfile, "w") as file:
        file.write(code)

    # Debugging display
    for letter in lettersJson:
        letterSegments = lettersJson[letter]
        if show_all or letter in show_letters: plotCharacter(letterSegments)

    return outfile

# Methods for GUI
def runFromGUI(erodeAmount, skelMethod, compressionAmount, fontFile, preview=False, previewChar=None):
    # Drop in method to alter configs and run
    global font
    global show_all
    global show_letters
    global erode_iterations
    global show_letters
    global line_combine_degree_threshold
    global skeletonize_method_index

    if not previewChar or previewChar == "":
        previewChar = "A"

    # Set general stuff for GUI mode
    show_letters = []
    show_all = False

    # Set stuff using options from GUI mode
    font = ImageFont.truetype(fontFile, 400)
    skeletonize_method_index = skeletonize_methods.index(skelMethod)
    erode_iterations = erodeAmount
    line_combine_degree_threshold = compressionAmount

    # Finally
    if preview:
        # Preview returns lines
        return getProcessedLinesForInputChars(previewChar, True)
    
    else:
        # Otherwise returns file path
        return getLinesForAlphabetAndExport()



# Run only if this is being run by hand, as an import it won't run
if __name__ == "__main__":
    getLinesForAlphabetAndExport()
