from typing import Dict
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel,Field
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from pymongo import MongoClient
import motor.motor_asyncio
from datetime import datetime
import io
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# MongoDB setup
from urllib.parse import quote_plus

username = quote_plus("barmasai")
password = quote_plus("Adm@3154")
MONGO_URI = f"mongodb+srv://{username}:{password}@school.qouaane.mongodb.net/?retryWrites=true&w=majority&appName=school"
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client['ai_making']  # Database name
features_collection = db['features']  # Collection name for features
data_collection = db['data']  # Collection name for storing datasets

model = None
preprocessor = None

class Features(BaseModel):
    status_of_existing_checking_account: str
    credit_history: str
    purpose: str
    savings_account_and_bonds: str
    present_employment_since: str
    personal_status_and_sex: str
    other_debtors_or_guarantors: str
    property: str
    other_installment_plans: str
    housing: str
    job: str
    telephone: str
    foreign_worker: str
    duration_in_month: float
    credit_amount: float
    installment_rate_in_percentage_of_disposable_income: float
    present_residence_since: float
    age_in_years: float
    number_of_existing_credits_at_this_bank: float
    number_of_people_being_liable_to_provide_maintenance_for: float


class FeatureConfig(BaseModel):
    numeric_features: list
    categorical_features: list

class PredictionResponse(BaseModel):
    default_probability: float
    creditworthiness: str
    model_confidence: float
    ai_credit_score: float
    recommendations: list
    model_accuracy: float
    feature_weights: dict
# Define the path to your models
MODEL_PATH = "models"

# Initialize the models dictionary
models: Dict[str, Dict[str, object]] = {}

@app.post("/upload_dataset")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        # Read the file content into a Pandas DataFrame
        contents = await file.read()
        data = pd.read_csv(io.BytesIO(contents))
        
        # Save the dataset to MongoDB
        data_dict = data.to_dict(orient='records')
        await data_collection.insert_many(data_dict)
        
        logger.info(f"Dataset {file.filename} uploaded successfully.")
        return {"message": "Dataset uploaded successfully.", "file_name": file.filename}
    except Exception as e:
        logger.error(f"Error uploading dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/set_features")
async def set_features(config: FeatureConfig):
    try:
        # Create a new feature configuration document
        config_dict = config.dict()
        # Save the feature configuration to MongoDB
        result = await features_collection.update_one(
            {"_id": 1},  # Example ID, you might want to use a unique ID or a different mechanism
            {"$set": config_dict},
            upsert=True
        )
        if result.matched_count == 0:
            logger.info("New feature configuration created.")
            return {"message": "New feature configuration created."}
        else:
            logger.info("Feature configuration updated.")
            return {"message": "Feature configuration updated."}
    except Exception as e:
        logger.error(f"Error setting feature configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_features/{model_number}")
async def get_features(model_number: int):
    try:
        # Fetch the feature configuration from MongoDB based on the model_number
        config = await features_collection.find_one({"_id": model_number})
        if not config:
            raise HTTPException(status_code=404, detail="Configuration not found.")
        
        logger.info(f"Feature configuration for model number {model_number} retrieved successfully.")
        return config
    except Exception as e:
        logger.error(f"Error retrieving feature configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train12")
async def train(model_number: int):
    try:
        # Load configuration
        config = await features_collection.find_one({"_id": model_number})
        if not config:
            raise HTTPException(status_code=404, detail="Configuration not found.")
        
        # Load data from MongoDB
        data = pd.DataFrame(await data_collection.find().to_list(length=1000))
        if data.empty:
            raise HTTPException(status_code=404, detail="No data available for training.")
        
        # Define features and target
        X = data.drop('creditability', axis=1)
        y = data['creditability']
        
        numeric_features = config['numeric_features']
        categorical_features = config['categorical_features']

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', LogisticRegression())
        ])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = pipeline.predict(X_test)
        model_accuracy = accuracy_score(y_test, y_pred)

        if model_accuracy < 0.8:
            logger.warning(f"Model accuracy is below 80%: {model_accuracy}. The model was not saved.")
            return {
                "message": "Model accuracy is below 80%. The model was not saved.",
                "model_accuracy": model_accuracy
            }

        # Ensure the 'models' directory exists
        models_dir = 'models'
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        # Save model and preprocessor
        model_path = os.path.join(models_dir, 'credit_model.pkl')
        preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')
        
        joblib.dump(pipeline.named_steps['model'], model_path)
        joblib.dump(pipeline.named_steps['preprocessor'], preprocessor_path)
        
        # Extract feature names and weights
        num_features = preprocessor.transformers_[0][1].named_steps['scaler'].get_feature_names_out()
        cat_features = preprocessor.transformers_[1][1].get_feature_names_out()
        feature_names = num_features.tolist() + cat_features.tolist()

        # Check if feature names match with model's coefficients
        model = pipeline.named_steps['model']
        weights = model.coef_.flatten()

        if len(feature_names) != len(weights):
            logger.error(f"Feature names and weights length mismatch: {len(feature_names)} vs {len(weights)}")
            raise ValueError(f"Feature names and weights length mismatch: {len(feature_names)} vs {len(weights)}")

        feature_weights = dict(zip(feature_names, weights))

        logger.info(f"Model trained and saved successfully with accuracy: {model_accuracy}.")
        return {
            "message": "Model trained and saved successfully.",
            "model_accuracy": model_accuracy,
            "feature_weights": feature_weights
        }
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict1")
async def predict(features: Features, model_name: str):
    if model_name not in models:
        model_path = os.path.join(MODEL_PATH, f"{model_name}.pkl")
        preprocessor_path = os.path.join(MODEL_PATH, f"{model_name}_preprocessor.pkl")

        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model file not found")

        if not os.path.exists(preprocessor_path):
            raise HTTPException(status_code=404, detail="Preprocessor file not found")

        try:
            # Load model and preprocessor
            model = joblib.load(model_path)
            preprocessor = joblib.load(preprocessor_path)

            # Store in models dictionary
            models[model_name] = {
                "model": model,
                "preprocessor": preprocessor
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading model or preprocessor: {str(e)}")

    model = models[model_name]["model"]
    preprocessor = models[model_name]["preprocessor"]

    # Convert features to DataFrame
    feature_values = pd.DataFrame([features.dict()])

    # Print out column names for debugging
    print("Input feature columns:", feature_values.columns.tolist())
    print("Expected columns:", preprocessor.get_feature_names_out())

    # Ensure all required columns are present
    missing_cols = set(preprocessor.get_feature_names_out()) - set(feature_values.columns)
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing_cols}")

    try:
        # Transform features
        feature_values_preprocessed = preprocessor.transform(feature_values)

        # Predict
        default_probability = model.predict_proba(feature_values_preprocessed)[0][1]
        creditworthiness = "good" if default_probability < 0.3 else "fair" if default_probability < 0.6 else "poor"
        model_confidence = 1 - abs(0.5 - default_probability)
        ai_credit_score = (1 - default_probability) * 850

        return {
            "default_probability": default_probability,
            "creditworthiness": creditworthiness,
            "model_confidence": model_confidence,
            "ai_credit_score": ai_credit_score,
            "recommendations": [
                "Maintain a low debt-to-income ratio.",
                "Limit the number of new credit accounts."
            ],
            "input_features": features.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    

# # @app.post("/train1")
# # async def train(model_number: int):
# #     try:
# #         # Load configuration
# #         config = await features_collection.find_one({"_id": model_number})
# #         if not config:
# #             raise HTTPException(status_code=404, detail="Configuration not found.")
        
# #         # Load data from MongoDB
# #         data = pd.DataFrame(await data_collection.find().to_list(length=1000))
# #         if data.empty:
# #             raise HTTPException(status_code=404, detail="No data available for training.")
        
# #         # Define features and target
# #         X = data.drop('creditability', axis=1)
# #         y = data['creditability']
        
# #         numeric_features = config['numeric_features']
# #         categorical_features = config['categorical_features']

# #         numeric_transformer = Pipeline(steps=[
# #             ('imputer', SimpleImputer(strategy='median')),
# #             ('scaler', StandardScaler())
# #         ])

# #         categorical_transformer = Pipeline(steps=[
# #             ('imputer', SimpleImputer(strategy='most_frequent')),
# #             ('onehot', OneHotEncoder(handle_unknown='ignore'))
# #         ])

# #         preprocessor = ColumnTransformer(
# #             transformers=[
# #                 ('num', numeric_transformer, numeric_features),
# #                 ('cat', categorical_transformer, categorical_features)
# #             ])

# #         pipeline = Pipeline(steps=[
# #             ('preprocessor', preprocessor),
# #             ('model', LogisticRegression())
# #         ])
        
# #         # Split data
# #         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
# #         # Train model
# #         pipeline.fit(X_train, y_train)
        
# #         # Evaluate model
# #         y_pred = pipeline.predict(X_test)
# #         model_accuracy = accuracy_score(y_test, y_pred)

# #         if model_accuracy < 0.8:
# #             logger.warning(f"Model accuracy is below 80%: {model_accuracy}. The model was not saved.")
# #             return {
# #                 "message": "Model accuracy is below 80%. The model was not saved.",
# #                 "model_accuracy": model_accuracy
# #             }

# #         # Save model and preprocessor
# #         joblib.dump(pipeline.named_steps['model'], 'credit_model.pkl')
# #         joblib.dump(pipeline.named_steps['preprocessor'], 'preprocessor.pkl')
        
# #         # Extract feature names and weights
# #         num_features = preprocessor.transformers_[0][1].named_steps['scaler'].get_feature_names_out()
# #         cat_features = preprocessor.transformers_[1][1].get_feature_names_out()
# #         feature_names = num_features.tolist() + cat_features.tolist()

# #         # Check if feature names match with model's coefficients
# #         model = pipeline.named_steps['model']
# #         weights = model.coef_.flatten()

# #         if len(feature_names) != len(weights):
# #             logger.error(f"Feature names and weights length mismatch: {len(feature_names)} vs {len(weights)}")
# #             raise ValueError(f"Feature names and weights length mismatch: {len(feature_names)} vs {len(weights)}")

# #         feature_weights = dict(zip(feature_names, weights))

# #         logger.info(f"Model trained and saved successfully with accuracy: {model_accuracy}.")
# #         return {
# #             "message": "Model trained and saved successfully.",
# #             "model_accuracy": model_accuracy,
# #             "feature_weights": feature_weights
# #         }
# #     except Exception as e:
# #         logger.error(f"Error training model: {str(e)}")
# #         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/train")
# async def train(model_number: int):
#     try:
#         config = await features_collection.find_one({"_id": model_number})
#         if not config:
#             raise HTTPException(status_code=404, detail="Configuration not found.")
        
#         # Load data from MongoDB
#         data = pd.DataFrame(await data_collection.find().to_list(length=1000))
#         if data.empty:
#             raise HTTPException(status_code=404, detail="No data available for training.")
        
#         logger.info("Data loaded successfully.")
        
#         # Check available columns
#         logger.info(f"Data columns: {data.columns.tolist()}")
        
#         # Check configuration features
#         numeric_features = config.get('numeric_features', [])
#         categorical_features = config.get('categorical_features', [])
#         logger.info(f"Numeric features: {numeric_features}")
#         logger.info(f"Categorical features: {categorical_features}")

#         # Ensure features exist in the data
#         missing_numeric_features = [col for col in numeric_features if col not in data.columns]
#         missing_categorical_features = [col for col in categorical_features if col not in data.columns]

#         if missing_numeric_features or missing_categorical_features:
#             raise HTTPException(status_code=400, detail=f"Missing columns in data: Numeric - {missing_numeric_features}, Categorical - {missing_categorical_features}")

#         X = data.drop('creditability', axis=1)
#         y = data['creditability']

#         numeric_transformer = Pipeline(steps=[
#             ('imputer', SimpleImputer(strategy='median')),
#             ('scaler', StandardScaler())
#         ])

#         categorical_transformer = Pipeline(steps=[
#             ('imputer', SimpleImputer(strategy='most_frequent')),
#             ('onehot', OneHotEncoder(handle_unknown='ignore'))
#         ])

#         global preprocessor
#         preprocessor = ColumnTransformer(
#             transformers=[
#                 ('num', numeric_transformer, numeric_features),
#                 ('cat', categorical_transformer, categorical_features)
#             ])

#         pipeline = Pipeline(steps=[
#             ('preprocessor', preprocessor),
#             ('model', LogisticRegression())
#         ])
        
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
#         pipeline.fit(X_train, y_train)
        
#         # Evaluate the model
#         from sklearn.metrics import accuracy_score
#         y_pred = pipeline.predict(X_test)
#         model_accuracy = accuracy_score(y_test, y_pred)

#         if model_accuracy < 0.8:
#             logger.warning(f"Model accuracy is below 80%: {model_accuracy}. The model was not saved.")
#             return {
#                 "message": "Model accuracy is below 80%. The model was not saved.",
#                 "model_accuracy": model_accuracy
#             }

#         # Save the model and preprocessor if accuracy is 80% or above
#         joblib.dump(pipeline.named_steps['model'], 'credit_model.pkl')
#         joblib.dump(pipeline.named_steps['preprocessor'], 'preprocessor.pkl')
        
#         # Get feature weights (coefficients) from the logistic regression model
#         feature_names = preprocessor.transformers_[0][1].named_steps['scaler'].get_feature_names_out() + \
#                         preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out()
#         weights = pipeline.named_steps['model'].coef_.flatten()
#         feature_weights = dict(zip(feature_names, weights))

#         logger.info(f"Model trained and saved successfully with accuracy: {model_accuracy}.")
#         return {
#             "message": "Model trained and saved successfully.",
#             "model_accuracy": model_accuracy,
#             "feature_weights": feature_weights
#         }
#     except Exception as e:
#         logger.error(f"Error training model: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))


  
@app.get("/load_model")
async def load_model_endpoint():
    global model, preprocessor
    try:
        model = joblib.load('credit_model.pkl')
        preprocessor = joblib.load('preprocessor.pkl')
        logger.info("Model loaded successfully.")
        return {"message": "Model loaded successfully."}
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Path where models are saved
MODEL_PATH = "."

class ModelResponse(BaseModel):
    model_name: str
    model_description: str


@app.get("/get_model/{model_name}", response_model=ModelResponse)
async def get_model(model_name: str):
    if model_name in models:
        return {
            "model_name": model_name,
            "model_description": models[model_name]["description"]
        }
    else:
        raise HTTPException(status_code=404, detail="Model not found")
@app.get("/models")
async def list_models():
    global models
    return {"models": list(models.keys())}
# Paths to your model and preprocessor files
model_path = r'C:\Users\Admin\Downloads\ai\models\credit_model.pkl'
preprocessor_path = r'C:\Users\Admin\Downloads\ai\models\preprocessor.pkl'

# Initialize the models dictionary
models = {}

# Load the model and preprocessor
try:
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    # Store in the models dictionary
    models['credit_model'] = {
        'model': model,
        'preprocessor': preprocessor
    }
    logger.info("Model and preprocessor loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model or preprocessor: {str(e)}")


    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
