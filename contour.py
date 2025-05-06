import cv2
import numpy as np

def find_contour(img_path):
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

    return spots, leaf_only_rgb