#!/usr/bin/env python3
"""
News Article Classification - Complete Project in Single File
=============================================================

This script contains the complete end-to-end machine learning pipeline:
1. Data Loading and Exploration
2. Data Preprocessing
3. Feature Engineering
4. Model Development and Training
5. Model Evaluation
6. Making Predictions on New Articles

Dataset Source: https://docs.google.com/spreadsheets/d/1m4YMfqQxo_DcbtzGqbfZitvJmytbWUE8qjixhHmtadk/edit?gid=1552269726#gid=1552269726

Usage:
    python complete_project.py

Requirements:
    - Dataset should be at: data/news_data.csv
    - All dependencies from requirements.txt installed
"""

# ============================================================================
# PART 1: IMPORT ALL LIBRARIES
# ============================================================================

import pandas as pd
import numpy as np
import os
import pickle
import time
import re
import string
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Text processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Machine Learning models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Model evaluation
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Utilities
from scipy.sparse import hstack

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

print("="*80)
print("ALL LIBRARIES IMPORTED SUCCESSFULLY!")
print("="*80)


# ============================================================================
# PART 2: LOAD AND EXPLORE DATASET
# ============================================================================

print("\n" + "="*80)
print("LOADING DATASET")
print("="*80)

df = pd.read_csv('data/news_data.csv')

print(f"Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())


# ============================================================================
# PART 3: DATA QUALITY ASSESSMENT
# ============================================================================

print("\n" + "="*80)
print("DATA QUALITY ASSESSMENT")
print("="*80)

# Check for missing values
missing_values = df.isnull().sum()
if missing_values.sum() > 0:
    print("\nMissing Values:")
    print(missing_values[missing_values > 0])
else:
    print("\n✓ No missing values found!")

# Check for duplicates
duplicate_count = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicate_count}")
if duplicate_count > 0:
    df = df.drop_duplicates()
    print(f"✓ Removed {duplicate_count} duplicate rows")

# Normalize category names
df['category'] = df['category'].str.strip().str.upper()

# Combine headline and description
df['headline'] = df['headline'].fillna('')
df['short_description'] = df['short_description'].fillna('')
df['combined_text'] = df['headline'] + ' ' + df['short_description']

print(f"\n✓ Dataset prepared: {df.shape[0]} articles, {df['category'].nunique()} categories")


# ============================================================================
# PART 4: CATEGORY DISTRIBUTION ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("CATEGORY DISTRIBUTION ANALYSIS")
print("="*80)

category_counts = df['category'].value_counts()
print(f"\nTotal categories: {df['category'].nunique()}")
print(f"\nTop 10 categories:")
print(category_counts.head(10))

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
top_15 = category_counts.head(15)
top_15.plot(kind='bar', ax=axes[0], color='steelblue')
axes[0].set_title('Top 15 Categories Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Category', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)

top_10_pct = (category_counts.head(10) / len(df) * 100)
top_10_pct.plot(kind='pie', ax=axes[1], autopct='%1.1f%%', startangle=90)
axes[1].set_title('Top 10 Categories (Percentage)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('')

plt.tight_layout()
os.makedirs('models/visualizations', exist_ok=True)
plt.savefig('models/visualizations/category_distribution.png', dpi=300, bbox_inches='tight')
print("\n✓ Category distribution visualization saved")
plt.close()


# ============================================================================
# PART 5: TEXT PREPROCESSING FUNCTIONS
# ============================================================================

print("\n" + "="*80)
print("TEXT PREPROCESSING")
print("="*80)

# Initialize text processing components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Clean text by removing HTML tags, special characters, and extra whitespace"""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

def preprocess_text(text, use_lemmatization=True, remove_stop=True):
    """Complete text preprocessing pipeline"""
    text = clean_text(text)
    tokens = word_tokenize(text)
    if remove_stop:
        tokens = [token for token in tokens if token not in stop_words]
    if use_lemmatization:
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

print("✓ Text preprocessing functions created!")


# ============================================================================
# PART 6: APPLY PREPROCESSING
# ============================================================================

print("\nApplying text preprocessing...")
print("This may take a few minutes for large datasets...")

df['cleaned_text'] = df['combined_text'].apply(
    lambda x: preprocess_text(x, use_lemmatization=True, remove_stop=True)
)

print("✓ Text preprocessing completed!")
print(f"\nSample original: {df['combined_text'].iloc[0][:150]}...")
print(f"\nSample cleaned: {df['cleaned_text'].iloc[0][:150]}...")


# ============================================================================
# PART 7: EXTRACT TEXTUAL FEATURES
# ============================================================================

print("\n" + "="*80)
print("FEATURE ENGINEERING")
print("="*80)

def extract_text_features(df):
    """Extract textual features from articles"""
    features = df.copy()
    features['char_count'] = features['cleaned_text'].str.len()
    features['word_count'] = features['cleaned_text'].str.split().str.len()
    features['avg_word_length'] = features['char_count'] / (features['word_count'] + 1)
    features['exclamation_count'] = features['cleaned_text'].str.count('!')
    features['question_count'] = features['cleaned_text'].str.count('?')
    return features

df_features = extract_text_features(df)
print("✓ Textual features extracted!")
print(f"\nFeature statistics:")
print(df_features[['char_count', 'word_count', 'avg_word_length']].describe())


# ============================================================================
# PART 8: TF-IDF VECTORIZATION
# ============================================================================

print("\n" + "="*80)
print("CREATING TF-IDF FEATURES")
print("="*80)

tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_tfidf = tfidf_vectorizer.fit_transform(df_features['cleaned_text'])
print(f"\n✓ TF-IDF matrix created: {X_tfidf.shape}")
print(f"  Features: {X_tfidf.shape[1]}")


# ============================================================================
# PART 9: PREPARE TARGET VARIABLE AND COMBINE FEATURES
# ============================================================================

# Encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_features['category'])

print(f"\n✓ Target variable encoded: {len(label_encoder.classes_)} categories")
print(f"\nCategory mapping (first 10):")
for i, label in enumerate(label_encoder.classes_[:10]):
    print(f"  {i}: {label}")
if len(label_encoder.classes_) > 10:
    print(f"  ... and {len(label_encoder.classes_) - 10} more")

# Combine TF-IDF with textual features
textual_features = df_features[['char_count', 'word_count', 'avg_word_length', 
                               'exclamation_count', 'question_count']].values

scaler = StandardScaler()
textual_features_scaled = scaler.fit_transform(textual_features)

X_combined = hstack([X_tfidf, textual_features_scaled])
print(f"\n✓ Combined features: {X_combined.shape}")


# ============================================================================
# PART 10: SPLIT DATA
# ============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

print("\n" + "="*80)
print("DATA SPLIT COMPLETED")
print("="*80)
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Training labels: {y_train.shape}")
print(f"Test labels: {y_test.shape}")


# ============================================================================
# PART 11: MODEL TRAINING FUNCTION
# ============================================================================

print("\n" + "="*80)
print("MODEL DEVELOPMENT")
print("="*80)

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Train a model and evaluate its performance"""
    print(f"\n{'='*60}")
    print(f"Training {model_name}...")
    print(f"{'='*60}")
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    y_pred = model.predict(X_test)
    
    # Calculate metrics (multi-class)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n{model_name} Results:")
    print(f"  Training Time: {training_time:.2f} seconds")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    return {
        'model': model,
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'training_time': training_time,
        'y_pred': y_pred
    }

print("✓ Model training function created!")


# ============================================================================
# PART 12: TRAIN MULTIPLE MODELS
# ============================================================================

results = {}

# 1. Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1, multi_class='ovr')
results['Logistic Regression'] = train_and_evaluate_model(
    lr_model, X_train, X_test, y_train, y_test, "Logistic Regression"
)

# 2. Naive Bayes
nb_model = MultinomialNB(alpha=1.0)
results['Naive Bayes'] = train_and_evaluate_model(
    nb_model, X_train, X_test, y_train, y_test, "Naive Bayes"
)

# 3. SVM (using linear kernel for speed)
svm_model = SVC(kernel='linear', probability=True, random_state=42)
results['SVM'] = train_and_evaluate_model(
    svm_model, X_train, X_test, y_train, y_test, "SVM"
)

# 4. Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=20)
results['Random Forest'] = train_and_evaluate_model(
    rf_model, X_train, X_test, y_train, y_test, "Random Forest"
)

# 5. XGBoost (convert sparse to dense)
X_train_dense = X_train.toarray() if hasattr(X_train, 'toarray') else X_train
X_test_dense = X_test.toarray() if hasattr(X_test, 'toarray') else X_test

xgb_model = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='mlogloss')
results['XGBoost'] = train_and_evaluate_model(
    xgb_model, X_train_dense, X_test_dense, y_train, y_test, "XGBoost"
)


# ============================================================================
# PART 13: COMPARE MODELS
# ============================================================================

comparison_data = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-Score': [],
    'Training Time (s)': []
}

for model_name, result in results.items():
    comparison_data['Model'].append(model_name)
    comparison_data['Accuracy'].append(result['accuracy'])
    comparison_data['Precision'].append(result['precision'])
    comparison_data['Recall'].append(result['recall'])
    comparison_data['F1-Score'].append(result['f1_score'])
    comparison_data['Training Time (s)'].append(result['training_time'])

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('F1-Score', ascending=False)

print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)
print(comparison_df.to_string(index=False))

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    comparison_df_sorted = comparison_df.sort_values(metric, ascending=True)
    ax.barh(comparison_df_sorted['Model'], comparison_df_sorted[metric], color='steelblue')
    ax.set_xlabel(metric, fontsize=12)
    ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    for i, v in enumerate(comparison_df_sorted[metric]):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('models/visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Model comparison visualization saved")
plt.close()

# Select best model
best_model_name = comparison_df.iloc[0]['Model']
best_model = results[best_model_name]['model']
print(f"\n✓ Best Model: {best_model_name}")


# ============================================================================
# PART 14: COMPREHENSIVE MODEL EVALUATION
# ============================================================================

print("\n" + "="*80)
print(f"EVALUATING BEST MODEL: {best_model_name}")
print("="*80)

y_pred = results[best_model_name]['y_pred']

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

# Classification report
print("\n" + "="*80)
print("CLASSIFICATION REPORT")
print("="*80)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


# ============================================================================
# PART 15: CONFUSION MATRIX
# ============================================================================

cm = confusion_matrix(y_test, y_pred)

# For multi-class, show top categories confusion matrix
top_categories = category_counts.head(10).index.tolist()
top_category_indices = [list(label_encoder.classes_).index(cat) for cat in top_categories if cat in label_encoder.classes_]

if len(top_category_indices) > 0:
    cm_top = cm[np.ix_(top_category_indices, top_category_indices)]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_top, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[label_encoder.classes_[i] for i in top_category_indices],
                yticklabels=[label_encoder.classes_[i] for i in top_category_indices])
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.ylabel('Actual', fontsize=12, fontweight='bold')
    plt.title(f'Confusion Matrix - Top {len(top_category_indices)} Categories', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('models/visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved for top {len(top_category_indices)} categories")
    plt.close()


# ============================================================================
# PART 16: SAVE MODELS AND VECTORIZERS
# ============================================================================

print("\n" + "="*80)
print("SAVING MODELS AND VECTORIZERS")
print("="*80)

os.makedirs('models', exist_ok=True)

# Save best model
with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Save TF-IDF vectorizer
with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

# Save scaler
with open('models/feature_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save label encoder
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("\nSaved files:")
print("  - models/best_model.pkl")
print("  - models/tfidf_vectorizer.pkl")
print("  - models/feature_scaler.pkl")
print("  - models/label_encoder.pkl")


# ============================================================================
# PART 17: PREDICTION FUNCTION FOR NEW ARTICLES
# ============================================================================

def predict_article_category(article_text):
    """
    Predict category for a new article
    
    Parameters:
    -----------
    article_text : str
        The news article text (headline + description)
    
    Returns:
    --------
    dict : Dictionary containing prediction and probabilities
    """
    # Preprocess
    cleaned_text = preprocess_text(article_text)
    
    # Extract textual features
    char_count = len(cleaned_text)
    word_count = len(cleaned_text.split())
    avg_word_length = char_count / (word_count + 1) if word_count > 0 else 0
    exclamation_count = cleaned_text.count('!')
    question_count = cleaned_text.count('?')
    
    # Transform using TF-IDF
    text_tfidf = tfidf_vectorizer.transform([cleaned_text])
    
    # Scale textual features
    textual_features = np.array([[char_count, word_count, avg_word_length, 
                                  exclamation_count, question_count]])
    textual_features_scaled = scaler.transform(textual_features)
    
    # Combine features
    features = hstack([text_tfidf, textual_features_scaled])
    
    # Make prediction
    prediction = best_model.predict(features)[0]
    category = label_encoder.inverse_transform([prediction])[0]
    
    # Get probabilities
    if hasattr(best_model, 'predict_proba'):
        probabilities = best_model.predict_proba(features)[0]
        prob_dict = {label_encoder.classes_[i]: probabilities[i] 
                    for i in range(len(label_encoder.classes_))}
        confidence = max(probabilities)
    else:
        prob_dict = None
        confidence = None
    
    return {
        'category': category,
        'confidence': confidence,
        'probabilities': prob_dict
    }

print("\n✓ Prediction function created!")


# ============================================================================
# PART 18: TEST PREDICTIONS
# ============================================================================

print("\n" + "="*80)
print("TESTING PREDICTIONS ON SAMPLE ARTICLES")
print("="*80)

sample_articles = [
    "Scientists discover new breakthrough in renewable energy technology that could revolutionize solar power efficiency.",
    "Local basketball team wins championship after thrilling overtime victory in the final game of the season.",
    "New study reveals benefits of meditation and mindfulness practices for mental health and stress reduction.",
    "Political leaders meet to discuss climate change policies and international cooperation agreements.",
    "Tech company announces revolutionary AI system that can understand and process natural language more accurately."
]

for i, article in enumerate(sample_articles, 1):
    result = predict_article_category(article)
    
    print(f"\nArticle {i}:")
    print(f"Text: {article[:80]}...")
    print(f"Predicted Category: {result['category']}")
    if result['probabilities']:
        sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"Top 3 Probabilities:")
        for cat, prob in sorted_probs:
            print(f"  {cat}: {prob:.2%}")
        print(f"Confidence: {result['confidence']:.2%}")
    print("-" * 80)


# ============================================================================
# PART 19: PROJECT SUMMARY
# ============================================================================

print("\n" + "="*80)
print("PROJECT SUMMARY")
print("="*80)

print(f"\n1. DATASET:")
print(f"   - Total articles: {len(df)}")
print(f"   - Categories: {df['category'].nunique()}")
print(f"   - Training samples: {len(y_train)}")
print(f"   - Test samples: {len(y_test)}")

print(f"\n2. BEST MODEL: {best_model_name}")
print(f"   - Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"   - F1-Score: {results[best_model_name]['f1_score']:.4f}")
print(f"   - Precision: {results[best_model_name]['precision']:.4f}")
print(f"   - Recall: {results[best_model_name]['recall']:.4f}")

print(f"\n3. KEY INSIGHTS:")
print(f"   - Model can classify articles into {len(label_encoder.classes_)} categories")
print(f"   - Best performing algorithm: {best_model_name}")
print(f"   - Model saved and ready for production use")

print(f"\n4. FILES CREATED:")
print(f"   - models/best_model.pkl")
print(f"   - models/tfidf_vectorizer.pkl")
print(f"   - models/feature_scaler.pkl")
print(f"   - models/label_encoder.pkl")
print(f"   - models/visualizations/*.png")

print("\n" + "="*80)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("="*80)

