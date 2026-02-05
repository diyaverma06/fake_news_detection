import streamlit as st
import pickle
import os
import sys


# Add src to path to import preprocess
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from preprocess import preprocess_text

# Set page config
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .real {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .fake {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    
    with open(os.path.join(models_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
        
    with open(os.path.join(models_dir, 'model_lr.pkl'), 'rb') as f:
        lr_model = pickle.load(f)
        
    with open(os.path.join(models_dir, 'model_svm.pkl'), 'rb') as f:
        svm_model = pickle.load(f)
        
    return vectorizer, lr_model, svm_model

def main():
    st.title("📰 Fake News Detection System")
    st.markdown("Enter a news headline or article snippet below to check if it's likely **Real** or **Fake**.")
    
    # Load models
    try:
        vectorizer, lr_model, svm_model = load_models()
    except FileNotFoundError:
        st.error("Models not found! Please ensure training has been run.")
        return

    # User Input
    news_text = st.text_area("News Text", height=150, placeholder="Type or paste the news article here...")
    
    # Model Selection
    model_choice = st.selectbox("Select Model", ["Logistic Regression", "Support Vector Machine (SVM)"])
    
    if st.button("Analyze News"):
        if news_text.strip():
            # Preprocess
            processed_text = preprocess_text(news_text)
            
            # Vectorize
            vectorized_text = vectorizer.transform([processed_text])
            
            # Predict
            if model_choice == "Logistic Regression":
                prediction = lr_model.predict(vectorized_text)[0]
                proba = lr_model.predict_proba(vectorized_text)[0]
                confidence = max(proba)
            else:
                prediction = svm_model.predict(vectorized_text)[0]
                proba = svm_model.predict_proba(vectorized_text)[0]
                confidence = max(proba)
            
            # Display Result
            st.divider()
            
            result_class = "real" if prediction == "REAL" else "fake"
            icon = "✅" if prediction == "REAL" else "⚠️"
            
            st.markdown(f"""
                <div class="result-card {result_class}">
                    {icon} Prediction: {prediction} News
                </div>
            """, unsafe_allow_html=True)
            
            st.info(f"Confidence Score: {confidence:.2%}")
            
            with st.expander("See Processed Text"):
                st.write(processed_text)
                
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main()
