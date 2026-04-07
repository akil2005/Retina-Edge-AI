import cv2
import numpy as np

def validate_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False, "ERROR: Image not found", 0.0

    # Convert to grayscale for mathematical analysis
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # need to add a gausian blur to reduce noise before edge detection
    gray = cv2.GaussianBlur(gray, (5,5),0)

    # Calculate edge sharpness using the Laplacian variance method
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < 100:
        return False, "REJECTED: Blurry", blur_score

    # Check if the overall lighting is within a diagnostic range
    brightness = np.mean(gray)
    if brightness < 40 or brightness > 220:
        return False, "REJECTED: Exposure issue", brightness

    # Measure the dynamic range to ensure lesions aren't washed out
    contrast = gray.std()
    if contrast < 20:
        return False, "REJECTED: Low contrast", contrast

    # Weighted formula to generate a single 0-100 quality metric
    q_score = (0.5 * min(blur_score, 200)/2) + \
              (0.3 * min(contrast, 50)*2) + \
              (0.2 * (100 - abs(brightness-127)/1.27))
    
    return True, f"PASSED: Score {q_score:.1f}", q_score

def auto_crop_retina(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use Otsu's method to automatically separate the eye from the black background
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Use a circular brush to rub out small noise speckles in the background
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Trace the boundaries of all white objects found in the mask
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Select the largest object (the retina) and ignore smaller artifacts
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        if perimeter == 0: 
            return image
        
        # Verify if the detected shape is round enough to be a human eye
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        x, y, w, h = cv2.boundingRect(cnt)
        img_h, img_w = image.shape[:2]

        # Apply a tight margin for clear eyes and a wide margin for ambiguous ones
        if circularity > 0.7:
            margin = int(max(w, h) * 0.10)
        elif 0.4 < circularity <= 0.7:
            margin = int(max(w, h) * 0.25)
        else:
            return image

        # Slice the image array using the calculated safety boundaries
        y1, y2 = max(0, y - margin), min(img_h, y + h + margin)
        x1, x2 = max(0, x - margin), min(img_w, x + w + margin)
        
        return image[y1:y2, x1:x2]
    
    return image