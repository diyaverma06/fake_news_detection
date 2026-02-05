import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure necessary NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except Exception as e:
        print(f"Warning: Failed to download punkt: {e}")

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        print(f"Warning: Failed to download punkt_tab: {e}")

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    try:
        nltk.download('stopwords', quiet=True)
    except Exception as e:
        print(f"Warning: Failed to download stopwords: {e}")

def preprocess_text(text):
    """
    Preprocesses the input text for the fake news detection model.
    Steps:
    1. Lowercasing
    2. Removing punctuation
    3. Tokenization
    4. Removing stopwords
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Reconstruct text
    processed_text = " ".join(filtered_tokens)
    
    return processed_text

if __name__ == "__main__":
    # Simple test
    sample_text = "Breaking: Aliens have landed in Times Square!"
    print(f"Original: {sample_text}")
    print(f"Processed: {preprocess_text(sample_text)}")
