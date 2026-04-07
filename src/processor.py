import cv2
import numpy as np

def enhance_clinical_image(image):
    #Extract the green channel Index
    green_channel = image[:, :, 1]
    
    clahe = cv2.createCLAHE(clipLimit = 2.0, titleGridSize = (8,8))
    enhanced_green = clahe.apply(green_channel)

    # Merge the enhanced green channel back with the original red and blue channels
    # stack the channels back together for the models compatibility

    final_image = cv2.merge([enhanced_green, enhanced_green, enhanced_green])
    return final_image