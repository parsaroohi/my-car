# libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import reshape
from numpy.core.numeric import zeros_like
from numpy.lib.function_base import average
from numpy.lib.type_check import imag

# read an image
img = cv2.imread('test_image.jpg')
# make a copy of image
lane_img = np.copy(img)


def make_coordinates(image, line_parameters):
    # make coordinates
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    # finding the average line(with average slope)
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        # creating grade 1 polynomial
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            right_fit.append((slope, intercept))
        else:
            left_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

# convert image to gray scale, decrease the noise and detect the edges


def canny(img):
    # convert image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # decrease the noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # edge detection
    canny = cv2.Canny(blur, 50, 150)
    return canny


def display_lines(image, lines):
    # to check detected lines of the image
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            # coordinates of lines
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

# show a region with a white-filled triangle


def region_of_interest(img):
    height = img.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    # using bitwise operand to combine two image and make a new one
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# use a function to detect the edges of the image
canny_image = canny(lane_img)
cropped_image = region_of_interest(canny_image)
# Hogh Transform
lines = cv2.HoughLinesP(cropped_image, 2,
                        np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
line_image = display_lines(lane_img, lines)
# blend two images with specific weights
combo_image = cv2.addWeighted(lane_img, 0.8, line_image, 1, 1)
average_lines = average_slope_intercept(lane_img, lines)
# to show average lines in first image
line_image = display_lines(lane_img, average_lines)
combo_image = cv2.addWeighted(lane_img, 0.8, line_image, 1, 1)
cv2.imshow('lines', line_image)
cv2.imshow('combo', combo_image)
# use matplotlib module to show the image and information of it
# show dimentional of the image (x, y)
plt.imshow(canny)
plt.show()
# show the image
cv2.imshow('canny', canny)
cv2.imshow('region', region_of_interest(canny))
cv2.imshow('cropped image', cropped_image)
cv2.waitKey(0)

# _____________________________________________________________
# work with videos
# Notice: for capturing the video have to comment lines 83 to 105
cap = cv2.VideoCapture('test2.mp4')
while (cap.isOpened()):
    _, frame = cap.read()
    # use a function to detect the edges of the image
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    # Hogh Transform
    lines = cv2.HoughLinesP(cropped_image, 2,
                            np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    # blend two images with specific weights
    average_lines = average_slope_intercept(frame, lines)
    # to show average lines in first image
    line_image = display_lines(frame, average_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow('combo', combo_image)
    if cv2.waitKey(1) == ord('q'):
        break
    # cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()
