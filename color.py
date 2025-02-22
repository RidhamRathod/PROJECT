import cv2
import numpy as np
from sklearn.cluster import KMeans

def determine_mood(image_path, n_colors=5):
    # Load image in color mode
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Error: Unable to load image. Check the file path.")

    # Convert from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape image to a 2D array of pixels
    pixels = image.reshape(-1, 3)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)

    # Get dominant colors
    dominant_colors = kmeans.cluster_centers_

    # Determine mood
    warm, cool, neutral = 0, 0, 0

    for r, g, b in dominant_colors:
        if r > g and r > b:
            warm += 1
        elif g > r and g > b or b > r and b > g:
            cool += 1
        else:
            neutral += 1

    if warm > cool and warm > neutral:
        mood = "Energetic and Exciting – Warm colors evoke passion, energy, and enthusiasm."
    elif cool > warm and cool > neutral:
        mood = "Calm and Relaxing – Cool colors promote trust, peace, and stability."
    elif neutral > warm and neutral > cool:
        mood = "Minimalistic and Neutral – Neutral tones create a sense of balance and sophistication."
    else:
        mood = "Balanced and Harmonious – A mix of warm, cool, and neutral colors creates visual balance."

    return mood