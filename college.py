# Install required libraries if not installed:
# pip install pandas scikit-learn transformers torch tqdm scipy nltk matplotlib

import pandas as pd
import nltk
import warnings
import matplotlib.pyplot as plt
from tqdm import tqdm
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

warnings.filterwarnings('ignore')
nltk.download('stopwords')
tqdm.pandas()

# ====================
# Text Preprocessing
# ====================
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = text.lower().split()
    return ' '.join([word for word in tokens if word.isalpha() and word not in stop_words])

# ====================
# Load Dataset
# ====================
train_data = pd.read_csv('feedback_train.csv')
test_data = pd.read_csv('feedback_test.csv')

train_data['cleaned_text'] = train_data['Feedback'].apply(preprocess_text)
test_data['cleaned_text'] = test_data['Feedback'].apply(preprocess_text)

# ====================
# TF-IDF + Logistic Regression
# ====================
tfidf = TfidfVectorizer(max_features=5000)
X_train = tfidf.fit_transform(train_data['cleaned_text'])
X_test = tfidf.transform(test_data['cleaned_text'])

y_train = train_data['Sentiment']
y_test = test_data['Sentiment']

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

print("\n=== College Feedback Sentiment Analysis ===")
print("\nLogistic Regression Results:")
print(classification_report(y_test, lr_predictions))

# ====================
# RoBERTa Sentiment Analysis
# ====================
print("\nLoading RoBERTa model...")
model_name = 'cardiffnlp/twitter-roberta-base-sentiment'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def get_roberta_sentiment(text):
    text = text.replace('@', '').replace('#', '').replace('RT', '')
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True)
    output = model(**encoded_input)
    scores = softmax(output.logits.detach().numpy()[0])
    labels = ['negative', 'neutral', 'positive']
    return labels[scores.argmax()]

test_data['bert_sentiment'] = test_data['Feedback'].progress_apply(get_roberta_sentiment)

print("\nRoBERTa Sentiment Analysis Results:")
print(classification_report(y_test, test_data['bert_sentiment']))

# ====================
# Category-wise Sentiment Summary
# ====================
print("\n=== Category-wise Sentiment Distribution ===")
category_summary = test_data.groupby(['Category', 'bert_sentiment']).size().unstack(fill_value=0)
print(category_summary)
category_summary.to_csv('category_sentiment_summary.csv')

# ====================
# Save Predictions
# ====================
test_data['lr_predictions'] = lr_predictions
test_data.to_csv('feedback_predictions.csv', index=False)
print("\nPredictions saved to 'college_feedback_predictions.csv'.")

# ====================
# Confusion Matrices
# ====================
def plot_conf_matrix(true, pred, title, cmap):
    cm = confusion_matrix(true, pred, labels=['positive', 'neutral', 'negative'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['positive', 'neutral', 'negative'])
    disp.plot(cmap=cmap)
    plt.title(title)
    plt.show()

plot_conf_matrix(y_test, lr_predictions, "Confusion Matrix - Logistic Regression", plt.cm.Blues)
plot_conf_matrix(y_test, test_data['bert_sentiment'], "Confusion Matrix - RoBERTa Model", plt.cm.Oranges)
