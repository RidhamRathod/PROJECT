import pytesseract
import cv2
import spacy
import textstat
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import language_tool_python

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize NLTK Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Initialize LanguageTool for grammar checking
tool = language_tool_python.LanguageTool('en-US')

# Function to extract text from an image
def extract_text():
    image_path = r"C:\Users\Ridham\Desktop\AI-ADD-REVIEWER\images\poster 4.png"  # Directly using the image path
    img = cv2.imread(image_path)

    if img is None:
        return None, "Error: Could not load the image. Please check the file path."

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text.strip(), None

# Readability score (Flesch-Kincaid)
def evaluate_readability(text):
    score = textstat.flesch_reading_ease(text)
    if score > 80:
        return f"Score: {score:.2f} - ✅ Easy to read."
    elif score > 50:
        return f"Score: {score:.2f} - ⚠️ Moderate readability."
    else:
        return f"Score: {score:.2f} - ❌ Difficult to read."

# Sentiment intensity (VADER)
def evaluate_sentiment(text):
    sentiment = analyzer.polarity_scores(text)
    if sentiment['compound'] > 0.5:
        return f"Score: {sentiment['compound']:.2f} - ✅ Positive sentiment."
    elif sentiment['compound'] > -0.5:
        return f"Score: {sentiment['compound']:.2f} - ⚠️ Neutral sentiment."
    else:
        return f"Score: {sentiment['compound']:.2f} - ❌ Negative sentiment."

# Grammar accuracy
def evaluate_grammar(text):
    if not text.strip():
        return "Score: 0.00 - ❌ No text detected."
    
    matches = tool.check(text)
    error_rate = len(matches) / max(len(text.split()), 1)  # Avoid division by zero
    accuracy_score = max(1 - error_rate, 0)

    if accuracy_score > 0.8:
        return f"Score: {accuracy_score:.2f} - ✅ Strong grammar."
    elif accuracy_score > 0.5:
        return f"Score: {accuracy_score:.2f} - ⚠️ Moderate grammar accuracy."
    else:
        return f"Score: {accuracy_score:.2f} - ❌ Poor grammar."

# Call-to-action phrase detection
import re

def evaluate_call_to_action(text):
    call_to_action_keywords = ["buy", "now", "order", "get started", "shop", "click here"]
    
    count = sum(1 for phrase in call_to_action_keywords if re.search(rf"\b{re.escape(phrase)}\b", text, re.IGNORECASE))
    
    if count >= 2:
        return f"Score: {count} - ✅ Strong call to action."
    elif count == 1:
        return f"Score: {count} - ⚠️ Moderate call to action."
    else:
        return f"Score: {count} - ❌ Weak call to action."


# Extract text
text, error = extract_text()

if error:
    print(error)
else:
    print("\n--- Poster Evaluation Results ---")
    print(evaluate_readability(text))
    print(evaluate_sentiment(text))
    print(evaluate_grammar(text))
    print(evaluate_call_to_action(text))