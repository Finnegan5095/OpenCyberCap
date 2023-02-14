"""
Using the NanoCamera with CSI Camera
@author: Ayo Ayibiowu

"""
# from tkinter import Image
import cv2
import numpy as np
from PIL import Image
import io
import nanocamera as nano

def display_lines(frame, lines, line_color=(0, 255, 255), line_width=20):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image

# Isolate region of interest (Remove unwanted environmental objects/edges.)
def find_region_of_interest(canny_edges):
    height,width = canny_edges.shape
    mask = np.zeros_like(canny_edges)
    
    # Focusing on certain portion of image (this example is lower half).
    polygon = np.array([[
        (0, height * 1/2),
        (width, height * 1/2),
        (width, height),
        (0, height),
    ]], np.int32)
    
    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(canny_edges, mask)
    return (cropped_edges)

# Function that performs hough transform to find lines. 
# Will need to tune later.
def get_line_segements(filter_region_image):
    
    
    rho = 1; # Distance precision in pixels
    angle = np.pi / 180 # Angular precision in radians
    min_thresh = 10 # min num of votes.
    line_segments = cv2.HoughLinesP(filter_region_image, rho, angle, 
            min_thresh, np.array([]), minLineLength=10, maxLineGap=15)
    
    return line_segments

def make_points(frame, line):
    height,width,_ = frame.shape
    slope, intercept = line
    y1 = height  # bottom of the frame
    y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

    # bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]

def combine_line_segments(frame, line_segments):
        
    lane_lines_detected = []
    if line_segments is None:
        return lane_lines_detected
    
    height,width, _ = frame.shape # Review this (wtf is the underscore).
    left_fit = []
    right_fit = []
    
    boundary = 1/3
    left_region_boundary = width * (1 - boundary) # The left lane segment should be on the left third of the frame.
    right_region_boundary = width * boundary # The right lane segment should be on the right third of the frame.
    
    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2: # Skips line segments found that are vertical.
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))
                    
    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines_detected.append(make_points(frame, left_fit_average))
        
    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines_detected.append(make_points(frame, right_fit_average))

    # logging.debug('lane lines: %s' % lane_lines)  # [[[316, 720, 484, 432]], [[1009, 720, 718, 432]]]
    return lane_lines_detected

# Main Loop

if __name__ == '__main__':
    # Create the Camera instance
    # camera = nano.Camera(flip=0, width=640, height=480, fps=30)
    # For multiple CSI camera
    # camera_2 = nano.Camera(device_id=1, flip=0, width=1280, height=800, fps=30)
    # print('CSI Camera is now ready')
    
    # frame = camera.read()
    # cv2.imshow(frame, frame) 
    
    # Test sample lane image.
    # img = Image.open("Test_Lane_img2.jpg")
    img = Image.open("SampleImage.jpg")
    # img = Image.open("IMG-5863 (1).jpg")
    # img = Image.open("Test_Lane_img2.jpg")
    
    img.load()
    frame = np.asarray(img, dtype=np.uint8)

    # Apply Gaussian Blur
    kernel_size = (3, 3)
    gauss_image = cv2.GaussianBlur(frame, kernel_size, 0)
    
    # Defining color range to remove unwanted colors from images.
    """ Note in the future the ranges should be optimized based on the colors of the roads we are driving on.
        It may be worthwhile/possible to incorporate a calibration sequence in which the color ranges are calculated based on
        the first camera frame received from the camera. It also could be useful to perform this calibration sequence every
        x number of frames received from the camera to account for frequent changes in road color. 
    
    """
    lower_black = np.array([0, 0, 0])
    # upper_black = np.array([227, 100, 70])
    upper_black = np.array([227, 130, 90]) 
    
    # Rectangular kernel.
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    
    # Tansform gauss image to HSV color.
    hsv_image = cv2.cvtColor(gauss_image, cv2.COLOR_BGR2HSV)
    
    # Apply color range to HSV converted image.
    filtered_image = cv2.inRange(hsv_image, lower_black, upper_black)
    # cv2.imshow(filtered_image, frame)
    
    # Adds pixels to boundaries of image
    
    dilated_image = cv2.dilate(filtered_image, rectKernel, iterations=1) # Look at kernel parameter.
    # cv2.imshow(dilated_image, frame)
    
    # Canny edge detection
    low_thresh = 200
    high_thresh = 400
    canny_edges = cv2.Canny(dilated_image, low_thresh, high_thresh)
    # cv2.imshow(canny_edges, frame)
    
    # Calling function with canny_edges image we found earlier.
    filter_region_image = find_region_of_interest(canny_edges)
    # cv2.imshow(filter_region_image, frame)
     
    line_segments = get_line_segements(filter_region_image)
    # cv2.imshow(line_segments, frame)
    
    lane_lines = combine_line_segments(frame, line_segments)
    
    
    # Finding approximate middlepoint line.
    lane_x1 = lane_lines.__getitem__(0)[0][0]
    lane_x2 = lane_lines.__getitem__(1)[0][0]
    lane_y1 = lane_lines.__getitem__(0)[0][2]
    lane_y2 = lane_lines.__getitem__(1)[0][2]
    
    middlePoint_x = (lane_x1 + lane_x2) / 2
    middlePoint_x = round(middlePoint_x)
    middlePoint_y = (lane_y1 + lane_y2) / 2
    middlePoint_y = round(middlePoint_y)
    lane_lines.append([[middlePoint_x, 1080, middlePoint_y, 540]])
    
    
    #Overlaying image on the main frame.
    line_image = display_lines(frame, lane_lines)
    
    
        
    # display both the current frame and the fg masks
    cv2.imshow('Frame', frame)
    cv2.waitKey(0)
    cv2.imshow('Canny_Edges', canny_edges)
    cv2.waitKey(0)
    cv2.imshow('Filtered Region of Interest', filter_region_image)
    cv2.waitKey(0)
    cv2.imshow('Line Image', line_image) 
    cv2.waitKey(0)       
           
    # while True:
    #     try:
    #         # read the camera image
    #         frame = camera.read()
            
    #         # display the frame
    #         cv2.imshow("Video Frame", frame)
    #         if cv2.waitKey(25) & 0xFF == ord('q'):
    #             break
    #     except KeyboardInterrupt:
    #         break

    # close the camera instance
    # camera.release()

    # remove camera object
    # del camera
