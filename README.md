# News Article Classification Project

## Project Overview
This project aims to build a machine learning model that classifies news articles into predefined categories (such as sports, politics, technology, wellness, etc.) using Natural Language Processing (NLP) techniques. The project includes comprehensive text preprocessing, feature engineering, model development, and evaluation.

## Project Structure
```
news_article_classification/
├── 01_Data_Exploration_Preprocessing.ipynb  # Data loading, EDA, and text preprocessing
├── 02_Feature_Engineering.ipynb              # Feature extraction (TF-IDF, Word2Vec, etc.)
├── 03_Model_Development.ipynb                 # Model training (LR, NB, SVM, RF, XGBoost)
├── 04_Model_Evaluation.ipynb                   # Model evaluation and visualization
├── 05_Predict_New_Articles.ipynb              # Prediction on new articles
├── requirements.txt                            # Python dependencies
├── README.md                                   # Project documentation
├── DATASET_INSTRUCTIONS.md                     # Dataset download guide
├── download_dataset.py                         # Dataset download helper
├── data/                                       # Dataset directory
│   └── news_data.csv                          # Input dataset (to be added)
└── models/                                     # Saved models directory
    ├── vectorizer.pkl
    ├── models (various .pkl files)
    └── visualizations
```

## Installation

1. Clone or download this repository
2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

4. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Dataset

The dataset should be placed in the `data/` directory as `news_data.csv`. 

**Dataset Source:** [Google Sheets Link](https://docs.google.com/spreadsheets/d/1m4YMfqQxo_DcbtzGqbfZitvJmytbWUE8qjixhHmtadk/edit?gid=1552269726#gid=1552269726)

**To download the dataset:**
1. Open the Google Sheets link
2. Go to File → Download → Comma Separated Values (.csv)
3. Save as `news_data.csv` in the `data/` folder

## Usage

### Step 1: Data Exploration and Preprocessing
Run `01_Data_Exploration_Preprocessing.ipynb` to:
- Load and explore the dataset
- Analyze category distribution
- Perform text cleaning (remove HTML tags, special characters)
- Tokenization, stop word removal
- Lemmatization and stemming
- Save preprocessed data

### Step 2: Feature Engineering
Run `02_Feature_Engineering.ipynb` to:
- Extract textual features (word count, character count, etc.)
- Create TF-IDF vectors
- Generate Word2Vec embeddings (optional)
- Prepare features for modeling

### Step 3: Model Development
Run `03_Model_Development.ipynb` to:
- Split data into training and testing sets
- Train multiple models:
  - Logistic Regression
  - Naive Bayes
  - Support Vector Machine (SVM)
  - Random Forest
  - XGBoost
- Hyperparameter tuning
- Save trained models

### Step 4: Model Evaluation
Run `04_Model_Evaluation.ipynb` to:
- Evaluate model performance with multiple metrics
- Visualize confusion matrices
- Plot classification reports
- Analyze feature importance
- Generate word clouds by category
- Generate insights and recommendations

### Step 5: Predict New Articles
Run `05_Predict_New_Articles.ipynb` to:
- Load trained model and vectorizer
- Preprocess new article text
- Make category predictions
- Get prediction probabilities

## Features

The dataset contains:
- **category**: News article category (sports, politics, technology, wellness, etc.)
- **headline**: Article headline
- **short_description**: Brief description of the article
- **keywords**: Relevant keywords
- **links**: Article URL

## Models Implemented

1. **Logistic Regression**: Baseline linear model for multi-class classification
2. **Naive Bayes**: Probabilistic classifier (good for text)
3. **Support Vector Machine (SVM)**: Effective for high-dimensional text data
4. **Random Forest**: Ensemble tree-based model
5. **XGBoost**: Gradient boosting classifier

## Evaluation Metrics

- Accuracy
- Precision (macro and weighted averages)
- Recall (macro and weighted averages)
- F1-Score (macro and weighted averages)
- Confusion Matrix
- Classification Report

## Expected Results

- Model accuracy: 70-90% (depending on number of categories)
- F1-Score: 0.70-0.90
- Key insights on words/phrases affecting category classification
- Feature importance analysis

## Key Insights

Based on the analysis, common factors affecting classification include:
- Article headline and description content
- Specific keywords and phrases
- Article length and structure
- Category-specific language patterns

## Business Applications

1. **News Organizations**: Automatically categorize articles for better organization
2. **Content Aggregators**: Improve content recommendation systems
3. **Social Media Platforms**: Better content filtering and organization
4. **Research**: Analyze trends across different news categories

## Author
Student Project - News Article Classification

## License
This project is for educational purposes.

## Notes
- Make sure to run notebooks in sequence (01 → 02 → 03 → 04 → 05)
- Models and vectorizers are saved automatically after training
- Preprocessed data is saved for reuse between notebooks
- Text preprocessing may take time for large datasets

