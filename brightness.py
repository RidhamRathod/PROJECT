import cv2
import numpy as np

def evaluate_brightness(image_path):
    """
    Evaluates the brightness of an image using average pixel intensity.
    
    Parameters:
        image_path (str): Path to the image file.
    
    Returns:
        str: A message indicating the brightness level.
    """
    # Load the image
    image = cv2.imread(image_path)
    
    # Check if the image was loaded successfully
    if image is None:
        return "Error: Could not load the image. Please check the file path."

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute average brightness
    mean_brightness = np.mean(gray)

    # Interpret brightness level
    if mean_brightness < 50:
        return f"{mean_brightness:.1f} - ❌ Very Dark. Consider increasing exposure or lighting."
    elif mean_brightness < 120:
        return f"{mean_brightness:.1f} - ⚠️ Dim lighting. Image might lack clarity."
    elif mean_brightness < 200:
        return f"{mean_brightness:.1f} - ✅ Well-lit. The brightness is good."
    else:
        return f"{mean_brightness:.1f} - 🌟 Very Bright! Be careful of overexposure."
