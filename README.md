# Quora Duplicate Question Detection

A machine learning project to identify duplicate questions on Quora using Natural Language Processing (NLP) techniques. The project includes feature engineering, model training, and a Streamlit web application for real-time predictions.

## Project Overview

This project aims to detect whether a pair of questions are duplicates or not, which is a crucial task for maintaining quality content on Q&A platforms like Quora. The solution uses various NLP techniques and machine learning models to make predictions.

## Features

- Text preprocessing and cleaning
- Feature engineering including:
  - Basic features (length, word count)
  - Token features
  - Length-based features
  - Fuzzy matching features
  - Bag of Words (BoW) representation
- Machine learning models:
  - Random Forest Classifier
  - XGBoost Classifier
- Interactive web interface using Streamlit

## Project Structure

```
├── run1.ipynb                  # Initial EDA and model training
├── run2_advanced_features.ipynb # Advanced feature engineering
├── train.csv                   # Training dataset
├── model.pkl                   # Trained model
├── cv.pkl                      # CountVectorizer model
├── streamlit app/
│   ├── app.py                 # Streamlit web application
│   ├── helper.py             # Helper functions for preprocessing
│   ├── Procfile              # Heroku deployment file
│   ├── requirements.txt      # Dependencies for the app
│   └── setup.sh             # Setup script for deployment
```

## Installation

1. Clone the repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```
3. Install NLTK stopwords:
```python
import nltk
nltk.download('stopwords')
```

## Usage

### Running the Web Application

Navigate to the streamlit app directory and run:
```bash
cd "streamlit app"
streamlit run app.py
```

### Using the Application

1. Enter the first question in the first text input
2. Enter the second question in the second text input
3. Click on "Find" to get the prediction
4. The result will show whether the questions are duplicate or not

## Model Features

The model uses several features to detect duplicate questions:
- Question length (characters)
- Word count
- Common words between questions
- Total unique words
- Word share ratio
- Token-based features
- Length-based features
- Fuzzy matching scores

## Technologies Used

- Python
- NLTK
- Scikit-learn
- XGBoost
- Streamlit
- Pandas
- NumPy
- FuzzyWuzzy
- Beautiful Soup

## License

This project is licensed under the terms of the included LICENSE file.

## Author

SSarkar-1

## Acknowledgments

- Quora for providing the problem statement and dataset
- Streamlit for the awesome web framework