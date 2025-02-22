from flask import Flask, request, jsonify
import os
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensures the folder exists

# Import all five models
from aesthetic import predict_aesthetic_score
from blur import detect_blurriness
from brightness import evaluate_brightness
from contrast import calculate_rms_contrast
from color import determine_mood
from T_features import evaluate_sentiment, evaluate_call_to_action, evaluate_grammar, evaluate_readability

def generate_final_explanation(results):
    explanation = ""

    # Aesthetic Score Interpretation
    if results.get("Aesthetic Score") is not None:
        explanation += f"The image has a well-balanced aesthetic with a score of {results['Aesthetic Score']}. "
    else:
        explanation += "The aesthetic quality could not be determined. "

    # Color Psychology Interpretation
    color_psych = results.get("Color Psychology", "").lower()
    if color_psych:
        explanation += f"The colors are {color_psych}, which influence the viewer's emotional response. "
    else:
        explanation += "The color psychology analysis is unavailable. "

    # Contrast Analysis Interpretation
    contrast = results.get("Contrast Analysis", "").lower()
    if "low" in contrast or "poor" in contrast:
        explanation += "However, the contrast is low, which may reduce visual impact. "
    elif contrast:
        explanation += f"Contrast is {contrast}, making key elements stand out. "
    else:
        explanation += "Contrast analysis was inconclusive. "

    # Brightness and Blurriness Interpretation
    brightness = results.get("Brightness Analysis", "").lower()
    blurriness = results.get("Image Blurriness", "").lower()

    if "too high" in brightness or "too low" in brightness:
        explanation += f"Brightness is {brightness}, which may affect readability. "
    elif brightness:
        explanation += f"Brightness is {brightness}, maintaining a visually appealing balance. "

    if "blurry" in blurriness:
        explanation += "The image appears blurry, which may reduce its effectiveness. "
    elif blurriness:
        explanation += "The image is sharp and clear. "

    # Readability, Sentiment, Grammar, and CTA Interpretation
    readability = results.get("Readability", "").lower()
    sentiment = results.get("Sentiment Analysis", "").lower()
    grammar = results.get("Grammar", "").lower()
    call_to_action = results.get("Call To Action", "").lower()

    if "difficult" in readability:
        explanation += "The text is difficult to read and may need simplification. "
    elif readability:
        explanation += f"Readability is {readability}. "

    if "negative" in sentiment or "neutral" in sentiment:
        explanation += "The sentiment could be improved for better engagement. "
    elif sentiment:
        explanation += f"Sentiment is {sentiment}, making the message more engaging. "

    if "incorrect" in grammar:
        explanation += "There are grammar issues that need correction. "
    elif grammar:
        explanation += "Grammar is correct, ensuring professionalism. "

    if "weak" in call_to_action or "unclear" in call_to_action:
        explanation += "The call to action is weak and could be more compelling. "
    elif call_to_action:
        explanation += f"The call to action is {call_to_action}, encouraging user interaction. "

    # Final conclusion
    explanation += "Overall, the image conveys a strong visual message. "

    return explanation

@app.route("/upload", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded image
    image_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(image_path)

    try:
        # Process the image with all five models
        results = {
            "Aesthetic Score": predict_aesthetic_score(image_path),
            "Color Psychology": determine_mood(image_path),
            "Contrast Analysis": calculate_rms_contrast(image_path),
            "Brightness Analysis": evaluate_brightness(image_path),
            "Image Blurriness": detect_blurriness(image_path),
            "Readability": evaluate_readability(image_path),
            "Sentiment Analysis": evaluate_sentiment(image_path),
            "Grammar": evaluate_grammar(image_path),
            "Call To Action": evaluate_call_to_action(image_path),
        }

        # Generate the final explanation
        final_explanation = generate_final_explanation(results)

        return jsonify({"analysis_results": results, "final_explanation": final_explanation})

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Handle unexpected errors

if __name__ == "__main__":
    app.run(debug=True)