import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from preprocess import preprocess_text

def train_models():
    # Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '../data/news.csv')
    models_dir = os.path.join(current_dir, '../models')
    
    # Create models directory if not exists
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Load Data
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Preprocess
    print("Preprocessing text...")
    df['clean_text'] = df['text'].apply(preprocess_text)
    
    # Split Data
    X = df['clean_text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Vectorization
    print("Vectorizing...")
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression()
    lr_model.fit(X_train_tfidf, y_train)
    
    # Train SVM
    print("Training SVM...")
    svm_model = SVC(kernel='linear', probability=True) # Probability=True for potential confidence scores later
    svm_model.fit(X_train_tfidf, y_train)
    
    # Evaluate
    print("\n--- Evaluation ---")
    
    lr_pred = lr_model.predict(X_test_tfidf)
    print(f"Logistic Regression Accuracy: {accuracy_score(y_test, lr_pred)}")
    print(classification_report(y_test, lr_pred))
    
    svm_pred = svm_model.predict(X_test_tfidf)
    print(f"SVM Accuracy: {accuracy_score(y_test, svm_pred)}")
    print(classification_report(y_test, svm_pred))
    
    # Save Artifacts
    print("\nSaving models...")
    with open(os.path.join(models_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
        
    with open(os.path.join(models_dir, 'model_lr.pkl'), 'wb') as f:
        pickle.dump(lr_model, f)
        
    with open(os.path.join(models_dir, 'model_svm.pkl'), 'wb') as f:
        pickle.dump(svm_model, f)
        
    print(f"Models saved to {models_dir}")

if __name__ == "__main__":
    train_models()
