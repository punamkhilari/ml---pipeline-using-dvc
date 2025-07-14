# ml---pipeline-using-dvc
Tweet Sentiment Classifier – End-to-End ML Pipeline

This project builds a binary sentiment classification model using tweet text to predict happiness vs sadness.
It demonstrates a full ML pipeline using:

📦 scikit-learn for modeling

🛠 DVC (Data Version Control) for pipeline reproducibility

🐍 Clean modular Python scripts (src/)

📊 Model metrics tracking (metrics.json)

✅ YAML-based config and reproducible stages  

🚀 Pipeline Stages

| Stage                 | Description                                                                |
| --------------------- | -------------------------------------------------------------------------- |
| `data_ingestion`      | Downloads and filters tweet data (happiness & sadness only)                |
| `data_preprocessing`  | Cleans and preprocesses text (stopword removal, lemmatization)             |
| `feature_engineering` | Converts text to BoW (Bag of Words) features using `CountVectorizer`       |
| `model_building`      | Trains a classifier (e.g., Logistic Regression or similar)                 |
| `model_evaluation`    | Evaluates performance (accuracy, precision, recall, AUC) and saves metrics |
