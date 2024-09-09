# Multi-Label Text Classification with BERT and Streamlit

This project implements a multi-label text classification system using BERT (Bidirectional Encoder Representations from Transformers) with PyTorch Lightning. It includes a Streamlit web application for real-time predictions and feedback collection, which can be used to continuously improve the model.

## Project Architecture

The project consists of three main components:

1. **Data Preprocessing and Model Training**: Handles ARFF file parsing, data preparation, and model training using BERT and PyTorch Lightning.
2. **Streamlit Web Application**: Provides a user interface for making predictions and collecting feedback.
3. **Deployment Script**: Runs the Streamlit app and makes it accessible via ngrok.

### Flow of the Project

1. Parse ARFF data and prepare datasets
2. Train the BERT model using PyTorch Lightning
3. Optimize hyperparameters using Optuna
4. Save the trained model
5. Run the Streamlit app for predictions and feedback
6. Collect feedback and retrain the model periodically

## Code Implementation

### Major Libraries Used

- **arff**: For parsing ARFF (Attribute-Relation File Format) files
- **pandas & numpy**: For data manipulation and numerical operations
- **PyTorch & PyTorch Lightning**: For building and training the BERT model
- **transformers**: For BERT model and tokenizer
- **Optuna**: An AutoML Framework for hyperparameter optimization
- **Streamlit**: For creating the web application
- **pyngrok**: For exposing the Streamlit app to the internet

### Data Preprocessing

The project uses the `arff` library to parse the ARFF file containing the dataset. The parsed data is then converted into a pandas DataFrame for easier manipulation.

## Model Architecture
We use a BERT-based model for multi-label text classification. The model architecture is defined in the `BERTClassLightning` class, which extends pl.LightningModule.

## AutoML Framework: Optuna
Optuna is used for hyperparameter optimization. It automatically searches for the best hyperparameters (learning rate, dropout rate, and batch size) to minimize the validation loss.
Optuna works by defining an objective function that trains the model with different hyperparameters and returns a metric to be optimized (in this case, the validation loss). It then uses various sampling algorithms to explore the hyperparameter space efficiently.

## Streamlit Web Application
The Streamlit app (app.py) provides a user interface for making predictions and collecting feedback

## Deployment with ngrok
This script uses ngrok to make the Streamlit app accessible over the internet
