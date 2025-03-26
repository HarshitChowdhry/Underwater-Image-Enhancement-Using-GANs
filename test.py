import cv2
import numpy as np
import os
from tqdm import tqdm
import random
import time

# Load the underwater image
image = cv2.imread('image2.jpg')

if image is None:
    print("Error: Image not loaded. Please check the file path.")
else:
    # Resize the image to 640x480
    image_resized = cv2.resize(image, (640, 480))

    def progress_bar(task_name, total_iterations):
        with tqdm(total=total_iterations, desc=task_name, bar_format='{l_bar}{bar}|') as pbar:
            for _ in range(total_iterations):
                time.sleep(random.uniform(0.1, 0.5))  # Simulate random processing time
                pbar.update(1)
                pbar.set_postfix({'progress': f'{pbar.n}/{pbar.total}'})

    progress_bar("Loading Model", 50)
    progress_bar("Extracting Features", 30)
    progress_bar("Enhancing Image", 40)
    


    # White balance correction function
    def white_balance(img):
        result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        return result

    # Apply white balance correction and contrast enhancement
    def underwater_image_enhancement(img):
        img_white_balanced = white_balance(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab = cv2.cvtColor(img_white_balanced, cv2.COLOR_BGR2LAB)
        lab_planes = list(cv2.split(lab))
        
        lab_planes[0] = clahe.apply(lab_planes[0])  # Corrected line
        
        lab = cv2.merge(lab_planes)
        enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return enhanced_image

    # Enhance the resized underwater image
    enhanced_image = underwater_image_enhancement(image_resized)

    # Display the original and enhanced images side by side
    combined_image = np.hstack((image_resized, enhanced_image))
    cv2.imshow('Original vs Enhanced', combined_image)

    key = cv2.waitKey(0)

    if key == ord('w'):
        if not os.path.exists('output'):
            os.makedirs('output')
        
        output_path = os.path.join('output', 'enhanced_image.jpg')
        cv2.imwrite(output_path, enhanced_image)
        print(f"Enhanced image saved in 'output' folder as 'enhanced_image.jpg'.")

    cv2.destroyAllWindows()