import cv2
import numpy as np
#from skimage.measure import block_reduce
#from matplotlib import pyplot as plt

# scipy.signal.find_peaks is the ideal solution to find peaks, but it relies on Scipy 1.1.0
#from scipy.signal import find_peaks

## peakutils is an alternative to find peaks in 1d array
#import peakutils

## detect_peaks is another function to find peaks in 1d array; only depends on Numpy; .py file downloaded
from detect_peaks import detect_peaks

# crop an image
def crop_image(img, lower_bound, upper_bound):
    # img_original_size = img.shape
    img_cropped = img[int(img.shape[0]*lower_bound):int(img.shape[0]*upper_bound),:]
    return img_cropped

# use color filter to show lanes in the image
def lane_filter(img, lower_lane_color, upper_lane_color):
    laneIMG = cv2.inRange(img, lower_lane_color, upper_lane_color)
    return laneIMG
        
# find the centers for two lines
def find_lane_centers(laneIMG_binary):
    # find peaks as the starting points of the lanes (left and right)
    vector_sum_of_lane_marks = np.sum(laneIMG_binary, axis = 0)
#    peaks, _ = find_peaks(vector_sum_of_lane_marks, distance=peaks_distance) 
#    peaks = peakutils.indexes(vector_sum_of_lane_marks, min_dist=peaks_distance)
    peaks = detect_peaks(vector_sum_of_lane_marks, mpd=peaks_distance)
    # we only use the first two peaks as the starting points of the lanes
    peaks = peaks[:2] 
    lane_center_left = peaks[0]
    lane_center_right = peaks[1]
    return lane_center_left, lane_center_right

# to find pixels/indices of one of the left and the right lane
# need to call twice, one for left line, and the other for right lane
def find_pixels_of_lane(laneIMG_binary, lane_center, window_size, width_of_laneIMG_binary):
    indices_nonzero = np.nonzero(laneIMG_binary[:,np.max([0, lane_center-window_size]):np.min([width_of_laneIMG_binary, lane_center+window_size])])
    x = indices_nonzero[0]
    y = indices_nonzero[1] + np.max([0,lane_center-window_size]) # shifted because we are using a part of laneIMG to find non-zero elements 
    return x, y

#For the color filter
lane_color = np.uint8([[[0,0,0]]])
lower_lane_color1 = lane_color
upper_lane_color1 = lane_color+40

# the distances of peaks should be tuned 
# the peaks indicate the possible center of left and right lanes
peaks_distance = 50

# size after downsampling
size_after_downsampling = (192, 32)

# this is the width of laneIMG
width_of_laneIMG_binary = size_after_downsampling[0]
height_of_laneIMG_binary = size_after_downsampling[1]

# use a window to find all pixels of the left lane and the right lane
# here, the window size is half as the distance between peaks
window_size = int(peaks_distance/2) 

# x_axis for polynomial fitting
number_points_for_poly_fit = 50
x_fitted = np.linspace(0, height_of_laneIMG_binary, number_points_for_poly_fit)
# polynomial fitting for lanes 
poly_order = 2


img=cv2.imread('03.jpg')
img = np.array(img)
# plt.figure(1)
# plt.imshow(img)

# cropping image
# img_original_size = img.shape
img_cropped = crop_image(img, 0.35, 0.6)
# plt.figure(2)
# plt.imshow(img_cropped)

# downsampling cropped image
img_downsampled = cv2.resize(img_cropped, size_after_downsampling, interpolation=cv2.INTER_LINEAR)
# plt.figure(3)
# plt.imshow(img_downsampled)

# color filtering image
# laneIMG = cv2.inRange(img_downsampled, lower_lane_color1, upper_lane_color1)
laneIMG = lane_filter(img_downsampled, lower_lane_color1, upper_lane_color1)
# making image to a binary representation
laneIMG_binary = laneIMG/255

# # find peaks as the starting points of the lanes (left and right)
# vector_sum_of_lane_marks = np.sum(laneIMG_binary, axis = 0)
# peaks, _ = find_peaks(vector_sum_of_lane_marks, distance=peaks_distance) 
# # we only use the first two peaks as the starting points of the lanes
# peaks = peaks[:2] 
# lane_center_left = peaks[0]
# lane_center_right = peaks[1]
lane_center_left, lane_center_right = find_lane_centers(laneIMG_binary)



# plt.figure(5)
# plt.plot(peaks, vector_sum_of_lane_marks[peaks], "x")
# plt.plot(vector_sum_of_lane_marks)


# use a window to find all pixels of the left lane and the right lane
# polynmial fitting for left lane 
# indices_left_lane = np.nonzero(laneIMG_binary[:,np.max([0,lane_center_left-window_size]):np.min([width_of_laneIMG_binary, lane_center_left+window_size])])
# x_left = indices_left_lane[0]
# y_left = indices_left_lane[1] + np.max([0,lane_center_left-window_size]) # shifted because we are using a part of laneIMG to find non-zero elements 
x_left, y_left = find_pixels_of_lane(laneIMG_binary, lane_center_left, window_size, width_of_laneIMG_binary)
w_left = np.polyfit(x_left, y_left, poly_order)
poly_fit_left = np.poly1d(w_left) # It is convenient to use poly1d objects for dealing with polynomials

# x_left_fitted = np.linspace(0, laneIMG_binary.shape[0])
# print('laneIMG_binary.shape[0]', laneIMG_binary.shape[0])
y_left_fitted = poly_fit_left(x_fitted) 

# polynmial fitting for right lane
# indices_right_lane = np.nonzero(laneIMG_binary[:,np.max([0, lane_center_right-window_size]):np.min([width_of_laneIMG_binary, lane_center_right+window_size])])
# x_right = indices_right_lane[0]
# y_right = indices_right_lane[1] + np.max([0, lane_center_right-window_size]) # shifted because we are using a part of laneIMG to find non-zero elements 
x_right, y_right = find_pixels_of_lane(laneIMG_binary, lane_center_right, window_size, width_of_laneIMG_binary)
w_right = np.polyfit(x_right, y_right, poly_order)
poly_fit_right = np.poly1d(w_right) # It is convenient to use poly1d objects for dealing with polynomials

# lane_right_fitted = np.zeros(laneIMG_binary.shape)
# x_right_fitted = np.linspace(0, laneIMG_binary.shape[0])
y_right_fitted = poly_fit_right(x_fitted)

# plt.figure(6)
# plt.plot(y_left_fitted, x_left, color='yellow', linewidth=4)
# plt.plot(y_right_fitted, x_right, color='yellow', linewidth=4)
# plt.imshow(img_downsampled)
# plt.show()

pts_left = np.array([y_left_fitted, x_fitted], np.int32).transpose()
pts_right = np.array([y_right_fitted, x_fitted], np.int32).transpose()
cv2.polylines(img_downsampled, [pts_left], False, (0,255,255), 1)
cv2.polylines(img_downsampled, [pts_right], False, (0,255,255), 1)
img_downsampled_zoomed = cv2.resize(img_downsampled, (0,0), fx=4, fy=4)

img_original_zoomed = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
cv2.imshow('image_original_zoomed', img_original_zoomed)
cv2.imshow('image_cropped', img_cropped)
cv2.imshow('image_downsampled', img_downsampled)
cv2.imshow('image_downsampled_zoomed', img_downsampled_zoomed)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()




