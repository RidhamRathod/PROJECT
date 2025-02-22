import cv2
import numpy as np

def calculate_rms_contrast(image_path):
    """Calculate RMS contrast for each RGB channel separately and return the average contrast with an explanation."""
    
    # Load the image in color (ensuring it has 3 channels)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  
    if image is None:
        raise ValueError("Error: Unable to load image. Check the file path.")

    # Convert from BGR (OpenCV default) to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Compute RMS contrast for each channel (R, G, B)
    rms_contrast = []
    for i in range(3):  # Loop through R, G, B channels
        channel = image[:, :, i]  # Extract channel
        mean_intensity = np.mean(channel)
        rms = np.sqrt(np.mean((channel - mean_intensity) ** 2))  # RMS formula
        rms_contrast.append(rms)

    # Average the RMS contrast of all three channels
    avg_rms_contrast = np.mean(rms_contrast)

    # Generate explanation based on contrast value
    if avg_rms_contrast < 30:
        explanation = f"{avg_rms_contrast:.1f} - The contrast is very low, making the image appear dull or washed out."
    elif 30 <= avg_rms_contrast < 60:
        explanation = f"{avg_rms_contrast:.1f} - The contrast is moderate, but some areas may lack distinction."
    elif 60 <= avg_rms_contrast < 90:
        explanation = f"{avg_rms_contrast:.1f} - The contrast is good, ensuring clear visibility of different elements."
    else:
        explanation = f"{avg_rms_contrast:.1f} - The contrast is very high, which can make the image look sharp but might cause harsh transitions."

    return explanation
