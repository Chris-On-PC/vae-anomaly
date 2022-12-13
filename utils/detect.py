import numpy as np
import cv2
import os


def convert_video(video_path):
    """
    """
    cap = load_video(video_path) # load video
    output_dir = create_output(video_path)
    frame_number = 1
    success, image = cap.read()
    temp_frame = None
    area = None
    while success:
        if area == None:
            area = detect_acquisition_area(image) # try detect acquisition areas
            if len(area) == 0 or area == None:
                continue # continue if none are detected

            # Only use channel captured in video
            if "Frontal" in video_path:
                area = area[0]

            elif "Lateral" in video_path and len(area)>1:
                area = area[1]
            else:
                area = None  
        else:
            
            # crop to area
            cropped_area = crop_image(image, area["x"], area["x"]+area["width"], area["y"], area["y"]+ area["height"])
            if temp_frame is None:
                temp_frame = cropped_area

            old_mean, new_mean = get_intensity_levels(temp_frame, cropped_area) # get some intesity values in area
            
            if new_mean != old_mean and new_mean != 0: # if the new value is not black and not equal to old value, save!
                start_frame  = frame_number
                # print(f"Start frame: {start_frame}")
                fullpath = os.path.join(output_dir, str(frame_number)+".png")
                cv2.imwrite(fullpath, cropped_area)

            temp_frame = cropped_area

        success, image = cap.read()
        frame_number += 1
    print("Done with video...")    
    return

def crop_image(image, startx, endx, starty, endy):
    """Utility for cropping images"""
    if (image.shape[:2][1] >= endx) and (image.shape[:2][0] >= endy):
        return image[starty:endy, startx:endx]
    else:
        raise ValueError


def load_video(video_path):
    cap = cv2.VideoCapture(os.path.abspath(video_path))
    return cap

def get_intensity_levels(old_frame, new_frame):
    old_patch = old_frame[int(old_frame.shape[0]/3): int(2*old_frame.shape[0]/3), int(old_frame.shape[1]/3): int(2*old_frame.shape[1]/3), :]
    new_patch = new_frame[int(new_frame.shape[0]/3): int(2*new_frame.shape[0]/3), int(new_frame.shape[1]/3): int(2*new_frame.shape[1]/3), :]

    old_mean = np.mean(old_patch)
    new_mean = np.mean(new_patch)
    return old_mean, new_mean
    
def create_output(video_path):
    top_folder = os.path.dirname(os.path.dirname(video_path))
    base_name = os.path.basename(video_path)
    # print(base_name)
    # filename, extension = os.path.split(base_name)
    # print(filename, extension)
    # print(os.path.split(base_name))
    
    new_folder = os.path.join(top_folder, base_name[:-4])
    print(new_folder)
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    
    return new_folder

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def detect_shapes( image, theshold_image, max_width = 0.8, max_height = 0.8, area = 5000):

    contours , hierarchy = cv2.findContours(theshold_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    squares = []

    # loop over the contours
    for cnt in contours:

        cnt_len = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
        

        if len(cnt) == 4 and cv2.contourArea(cnt) > area and cv2.isContourConvex(cnt):

            cnt = cnt.reshape(-1, 2)

            max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
            a = (cnt[1][1] - cnt[0][1])

            if max_cos < 0.1 and a < image.shape[1]*max_width and image.shape[0]*max_height:
                    squares.append(cnt)

    squares = [parse_coordinates(i.tolist()) for i in squares]

    return squares

def parse_coordinates(coordinates):
    max_x, max_y, min_x, min_y= 0,0, 99999, 999999
    for element in coordinates:
        x, y = element
        if x > max_x: max_x = x
        if y > max_y: max_y = y
        if x < min_x: min_x = x
        if y < min_y: min_y = y 

    return {"x": int(min_x), "y": int(min_y), "width": int(max_x-min_x), "height": int(max_y-min_y)}

def detect_acquisition_area(image):

    thresh = convert_image(image)
    shape = detect_shapes(image, thresh)
    return shape

def convert_image(image):

    
    if len(image.shape) == 3 :
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)[1]
    return  thresh
