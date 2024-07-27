"# AI-credit-scoring" 
# FastAPI Project

## Overview

This FastAPI project provides a web API for credit scoring and loan processing. It features endpoints for training machine learning models and making predictions based on user inputs.

## Features

- **Model Training**: Train and save machine learning models for credit scoring.
- **Prediction**: Make predictions using a trained model.
- **Configuration Management**: Fetch and manage feature configurations for model training.

## Requirements

- Python 3.8 or higher
- FastAPI
- Uvicorn
- scikit-learn
- pandas
- joblib

Install the required packages using `pip`:

```bash
pip install fastapi uvicorn scikit-learn pandas joblib
## Usage
Start the Server

Run the FastAPI server with Uvicorn:

bash
Copy code
uvicorn app:app --reload
This starts the server on http://127.0.0.1:8000.

Endpoints

Train Model

POST /train12

Request:

json
Copy code
{
  "model_name": "example_model"
}
Response:

json
Copy code
{
  "message": "Model trained and saved successfully.",
  "model_accuracy": 85.5,
  "feature_weights": {
    "feature1": 0.12,
    "feature2": -0.45
  }
}
Make Prediction

POST /predict1

Request:

json
Copy code
{
  "model_name": "example_model",
  "features": {
    "feature1": 1.2,
    "feature2": 3.4
  }
}
Response:

json
Copy code
{
  "default_probability": 85.0,
  "creditworthiness": "Good",
  "model_confidence": 90.0,
  "ai_credit_score": 850.0,
  "recommendations": [
    "Increase credit score",
    "Reduce debt"
  ],
  "model_accuracy": 85.5,
  "feature_weights": {
    "feature1": 0.12,
    "feature2": -0.45
  }
}
Configuration
Feature configurations for training models are stored in a MongoDB database. Ensure the database is properly set up and configured.

Database
MongoDB: Used for storing feature configurations and training data.
