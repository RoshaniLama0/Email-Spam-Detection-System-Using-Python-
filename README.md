# Email Spam Detector

## Project Description

This project is an Email Spam Detection System developed using Python and Machine Learning.

The system takes a message from the user and predicts whether the message is:

- SPAM
- NOT SPAM

The application provides a simple graphical user interface using Tkinter.

## Technologies Used

- Python
- Scikit-learn
- TF-IDF
- Multinomial Naive Bayes
- Tkinter
- Joblib

## How It Works

1. The user enters a message.
2. The message is converted into numerical features using TF-IDF.
3. The trained machine learning model analyzes the message.
4. The system predicts SPAM or NOT SPAM.
5. The application displays the prediction and confidence.

## Project Files

- `app.py` - Main GUI application
- `spam_detector.ipynb` - Jupyter Notebook containing the model training and testing
- `spam_model.pkl` - Saved trained machine learning model
- `tfidf.pkl` - Saved TF-IDF model
- `requirements.txt` - Required Python packages

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
