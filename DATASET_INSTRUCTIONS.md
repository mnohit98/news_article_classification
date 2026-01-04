# Dataset Download Instructions

## News Article Classification Dataset

This project uses the news article dataset from Google Sheets.

## Dataset Source

**Google Sheets Link:** [News Dataset](https://docs.google.com/spreadsheets/d/1m4YMfqQxo_DcbtzGqbfZitvJmytbWUE8qjixhHmtadk/edit?gid=1552269726#gid=1552269726)

## Download Methods

### Method 1: Manual Download (Recommended)

1. Open the Google Sheets link in your browser
2. Go to **File → Download → Comma Separated Values (.csv)**
3. Save the file as `news_data.csv` in the `data/` directory
4. Make sure the file has the following structure:
   - Column 1: `category` - The news category (e.g., WELLNESS, SPORTS, POLITICS, TECHNOLOGY)
   - Column 2: `headline` - The article headline
   - Column 3: `links` - Article URL
   - Column 4: `short_description` - Brief description of the article
   - Column 5: `keywords` - Relevant keywords

### Method 2: Direct Export URL

If the Google Sheet is publicly accessible, you can download it directly using:

```bash
curl -L "https://docs.google.com/spreadsheets/d/1m4YMfqQxo_DcbtzGqbfZitvJmytbWUE8qjixhHmtadk/export?format=csv&gid=1552269726" -o data/news_data.csv
```

Or use Python:

```python
import pandas as pd

url = "https://docs.google.com/spreadsheets/d/1m4YMfqQxo_DcbtzGqbfZitvJmytbWUE8qjixhHmtadk/export?format=csv&gid=1552269726"
df = pd.read_csv(url)
df.to_csv('data/news_data.csv', index=False)
```

### Method 3: Using the Download Script

Run the provided download script:

```bash
python download_dataset.py
```

## Dataset Structure

The dataset should contain:
- **category**: News article category (WELLNESS, SPORTS, POLITICS, TECHNOLOGY, etc.)
- **headline**: Article headline text
- **short_description**: Brief description of the article content
- **keywords**: Relevant keywords (may be used for feature engineering)
- **links**: Article URL

## Expected File Location

```
news_article_classification/
└── data/
    └── news_data.csv
```

## Verification

After downloading, verify the dataset:

```python
import pandas as pd

df = pd.read_csv('data/news_data.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Category distribution:\n{df['category'].value_counts()}")
```

## Notes

- The dataset may contain multiple categories
- Categories may need to be normalized (e.g., convert to lowercase)
- Combine `headline` and `short_description` for better classification
- Make sure the category column contains valid category labels

