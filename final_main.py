import cv2
from matplotlib import pyplot as plt
import os
import numpy as np
import random

# function to open images from the same folder
def pathList(path):
    files = [i for i in os.listdir(path) if i.lower().endswith(('png', 'jpg', 'jpeg'))]
    return [os.path.join(path, file) for file in files]

def segmentation(image_paths):
    
    path_list = []
    for img_path in image_paths:

        image = cv2.imread(img_path)                                # open the image
        image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)           # convert to RGB 
        image_hsv = cv2.cvtColor(image_rgb,cv2.COLOR_RGB2HSV)       # convert to HSV
        mask_greenyellow = cv2.inRange(image_hsv, (0 , 0, 0), (85, 255, 255))   # choose the HSV range to make the mask
        mask_brown = cv2.inRange(image_hsv, (15, 0, 0), (25, 255, 150))
        mask = cv2.bitwise_or(mask_greenyellow,mask_brown)
        kernel = np.ones((10, 10),np.uint8)                         # kernel necessary for the morphology
        mask_morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_3ch = cv2.cvtColor(mask_morph, cv2.COLOR_GRAY2BGR)     # convert the mask to 3 channels
        foreground = cv2.bitwise_and(image_rgb, mask_3ch)           # remove the background
        image_bgr = cv2.cvtColor(foreground,cv2.COLOR_RGB2BGR)
        
        # Save the image to a folder
        filename = os.path.basename(img_path)
        if not os.path.exists("segmented/"): 
            os.makedirs("segmented/") 
        save_path = os.path.join("segmented/", filename)
        cv2.imwrite(save_path, image_bgr)

    return image_bgr, path_list

#Detection based on color
def dominateColor(image):
    array_mean = []                                                         # initializing an array
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)                      # converting the image to HSV
    mask_mean = cv2.inRange(image, (2,2,2), (255,255,255))                  # applying a mask to ignore the background that is black
    hue = cv2.mean(image,mask = mask_mean)[0:1]                             # mean of hue calculated across the whole image
    saturation = cv2.mean(image,mask = mask_mean)[1:2]                      # mean of saturation calculated across the whole image
    value = cv2.mean(image,mask = mask_mean)[2:3]                           # mean of value calculated across the whole image
    array_mean.extend([hue, saturation, value])                             # values of the means are saved into the array
    return array_mean                                                       # the function returns the array with the means

def detectLeaf(img):                                                    
    image = cv2.imread(img)                                                 # reading the image
    kernel = np.ones((7, 7), np.uint8)                                      # defining the kernel for morphology operations
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)                            # converting the image to HSV

    mask_brown = cv2.inRange(hsv, (8, 60, 20), (30, 255, 200))              # defining a mask to isolate the brown color
    mask_yellow_green = cv2.inRange(hsv, (10, 39, 64), (86, 255, 255))      # defining a mask to isolate the yellow and green colors
    mask = cv2.bitwise_or(mask_yellow_green, mask_brown)                    # defining a mask that is the combination of the two color masks

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)                  # operation of closing
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)                   # operation of opening

    return mask

def colorDetection(image_paths):

    path_list = {}                                                                  # initializing a dictionary 

    for img_path in image_paths:                                                    # applying the detection for every image in the folder

        image = cv2.imread(img_path)                                                # reading the image
        mask = detectLeaf(img_path)                                                # applying the function detect_leaf
        image_masked = cv2.bitwise_and(image,image, mask=mask)                      # applying the mask
        image_masked_hsv = cv2.cvtColor(image_masked, cv2.COLOR_BGR2HSV)            # image masked is converted in RGB image
        array_hsv_mean = dominateColor(image_masked_hsv)                            # applying the function dominateColor
        
        # Checking the damage using the color and saving the images in two different folders wrt their status

        if (45 <= array_hsv_mean[0][0]<= 85) and (40 <= array_hsv_mean[1][0]<= 255) and (40 <= array_hsv_mean[2][0]<= 255):
            status = 'The leaf is healthy'
            filename = os.path.basename(img_path)
            if not os.path.exists("detected_healthy_leaves_color/"): 
                os.makedirs("detected_healthy_leaves_color/")
            save_path = os.path.join("detected_healthy_leaves_color/", filename)
            cv2.imwrite(save_path, image)
            
                        
        else: 
            status = 'The leaf is damaged'
            filename = os.path.basename(img_path)
            if not os.path.exists("detected_damaged_leaves_color/"): 
                os.makedirs("detected_damaged_leaves_color/")
            save_path = os.path.join("detected_damaged_leaves_color/", filename)
            cv2.imwrite(save_path, image)
            

        path_list[save_path]= status                                                # saving the results in a dictionary where each pair is composed by the path of the image as key and its status as value

    return path_list


def findContour(image_paths):

    path_list = {}

    for img_path in image_paths:

        image = cv2.imread(img_path)                                                        # import a segmented leaf image
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)                                        # convert the color to RGB
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                                      # convert the color to grayscale
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)                                        # convert thecolor to HSV
        leaf_mask = gray > 15                                                               # develop a mask for the leaf from the grayscale image
        leaf_mask_uint8 = (leaf_mask * 255).astype(np.uint8)                                # convert the mask to int8
        leaf_only = cv2.bitwise_and(gray, gray, mask=leaf_mask_uint8)                       # isolate the leaf on the grayscale image
        leaf_only_hsv = cv2.bitwise_and(hsv, hsv, mask=leaf_mask_uint8)                     # isolate the leaf on the HSV image
        leaf_only_rgb = cv2.bitwise_and(rgb, rgb, mask=leaf_mask_uint8)
        spot_mask = cv2.inRange(leaf_only_hsv, (10, 50, 0), (25, 255, 155))                 # develop a mask for the spot or damage
        spot_only = cv2.bitwise_and(leaf_only, leaf_only, mask=spot_mask)                   # isolate the spot or damage
        blur = cv2.GaussianBlur(spot_only, (3,3), 0)                                        # blur the spot image
        ret,threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)      # binarize the image using Otsu
        threshold_masked = cv2.bitwise_and(threshold, threshold, mask=spot_mask)            # isolate the binarized image using the spot mask

        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))                           # initialize a rectangular 3x3 kernel
        threshold_clean1 = cv2.morphologyEx(threshold_masked, cv2.MORPH_ERODE, kernel1)     # use morphology to erode the binarized image
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))                        # initialize an elipsoid 3x3 kernel
        threshold_clean2 = cv2.morphologyEx(threshold_clean1, cv2.MORPH_DILATE, kernel2)    # use morphology to dilate the binarized image
        # kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
        # threshold_clean3 = cv2.morphologyEx(threshold_clean2, cv2.MORPH_DILATE, kernel3)

        contours, hierarchy = cv2.findContours(threshold_clean2, 
                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)     # find the contours

        spots = 0                                                                           # start the contours counter
        for c in contours:
            area = cv2.contourArea(c)
            if area < 100:                                                                  # draw the contour if the area is <100
                cv2.drawContours(leaf_only_rgb, [c], -1, (0, 255, 0), 2)
                spots +=1                                                                   # count the contours
        
        leaf_only_bgr = cv2.cvtColor(leaf_only_rgb, cv2.COLOR_RGB2BGR)                                                                   
        
        if spots == 0:
            status = 'The leaf is healthy'
            filename = os.path.basename(img_path)
            if not os.path.exists("detected_healthy_leaves_contour/"): 
                os.makedirs("detected_healthy_leaves_contour/")
            save_path = os.path.join("detected_healthy_leaves_contour/", filename)
            cv2.imwrite(save_path, leaf_only_bgr)

        else:
            status = 'The leaf is damaged'
            filename = os.path.basename(img_path)
            if not os.path.exists("detected_damaged_leaves_contour/"): 
                os.makedirs("detected_damaged_leaves_contour/")
            save_path = os.path.join("detected_damaged_leaves_contour", filename)
            cv2.imwrite(save_path, leaf_only_bgr)

        path_list[save_path]= status

    return path_list





# folder_path = pathList("LeafClassifier-main/raw/segmented/Peach___Bacterial_spot")
# total_images = len(folder_path)
# accuracy_vector = []
# #segmented_list = segmentation(folder_path)
# status_leaves = colorDetection(folder_path)
# damaged_colordetected_imgs = len(pathList("detected_damaged_leaves_color/"))
# status_leaves_new = findContour(folder_path)
# damaged_contourdetected_imgs = len(pathList("detected_damaged_leaves_contour/"))

# if os.listdir("detected_healthy_leaves_color/"):
#      status_leaves_new2 = findContour(pathList("detected_healthy_leaves_color/"))
#      damaged_contourdetected_imgs2 = len(status_leaves_new2)

# accuracy_vector = [damaged_colordetected_imgs/total_images, damaged_contourdetected_imgs/total_images, (damaged_colordetected_imgs+damaged_contourdetected_imgs2)/total_images]    
# print(accuracy_vector)