<div align="center">

# 📰 Fake News Detection

**An NLP classifier that tells real news from fake — with a live Streamlit app.**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-154F5B?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

</div>

---

## ✨ What it does

Paste a news headline or article snippet, pick a model, and the app predicts whether it's likely **REAL** or **FAKE**.

## 🧠 How it works

1. **Preprocessing** — text is cleaned and normalized with NLTK (`src/preprocess.py`)
2. **Feature extraction** — articles are converted into **TF-IDF** vectors
3. **Models** — two classifiers are trained and compared (`src/train.py`):
   - Logistic Regression
   - Support Vector Machine (SVM)
4. **App** — a Streamlit interface lets you switch between models and test any text (`app.py`)

## 📁 Project structure

```
fake_news_detection/
├── app.py              # Streamlit web app
├── src/
│   ├── preprocess.py   # Text cleaning
│   └── train.py        # TF-IDF + model training & evaluation
├── data/               # News datasets (REAL / FAKE labels)
├── models/             # Trained vectorizer + LR and SVM models
└── requirements.txt
```

## 🚀 Run it locally

```bash
pip install -r requirements.txt
python src/train.py      # (optional) retrain the models
streamlit run app.py
```

## 🛠️ Tech stack

Python · scikit-learn · NLTK · pandas · NumPy · Streamlit

---

<div align="center">
Made with 💜 by <a href="https://github.com/diyaverma06">Diya Verma</a> · <a href="https://www.diyaverma.in">Portfolio</a>
</div>
