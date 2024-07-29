import io
import traceback
from typing import Dict
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging
import os
from pymongo import MongoClient
import motor.motor_asyncio
from urllib.parse import quote_plus

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# MongoDB setup
username = quote_plus("barmasai")
password = quote_plus("Adm@3154")
MONGO_URI = f"mongodb+srv://{username}:{password}@school.qouaane.mongodb.net/?retryWrites=true&w=majority&appName=school"
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client['ai_making']
features_collection = db['features']
data_collection = db['data']
model_collection = db['model_details_collection']
predictions_collection = db['predictions_collection']

# Initialize the models dictionary
models: Dict[str, Dict[str, object]] = {}

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
    model_name: str
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

MODEL_PATH = "models"

@app.post("/upload_dataset")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        data = pd.read_csv(io.BytesIO(contents))
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
        # Create a new feature configuration document with model name
        config_dict = config.dict()
        model_name = config_dict.pop('model_name')

        # Save the feature configuration to MongoDB
        result = await features_collection.update_one(
            {"model_name": model_name},
            {"$set": config_dict},
            upsert=True
        )
        if result.matched_count == 0:
            logger.info(f"New feature configuration created for model: {model_name}.")
            return {"message": f"New feature configuration created for model: {model_name}."}
        else:
            logger.info(f"Feature configuration updated for model: {model_name}.")
            return {"message": f"Feature configuration updated for model: {model_name}."}
    except Exception as e:
        logger.error(f"Error setting feature configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_features/{model_number}")
async def get_features(model_number: int):
    try:
        config = await features_collection.find_one({"_id": model_number})
        if not config:
            raise HTTPException(status_code=404, detail="Configuration not found.")
        logger.info(f"Feature configuration for model number {model_number} retrieved successfully.")
        return config
    except Exception as e:
        logger.error(f"Error retrieving feature configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train12")
async def train(model_name: str):
    try:
        # Fetch feature configuration using model_name
        config = await features_collection.find_one({"model_name": model_name})
        if not config:
            raise HTTPException(status_code=404, detail="Configuration not found.")
        
        # Load the data for training
        data = pd.DataFrame(await data_collection.find().to_list(length=1000))
        if data.empty:
            raise HTTPException(status_code=404, detail="No data available for training.")
        
        # Normalize column names
        data.columns = [col.replace('.', '_') for col in data.columns]
        
        logger.info(f"Normalized data columns: {data.columns.tolist()}")
        
        X = data.drop('creditability', axis=1)
        y = data['creditability']
        
        numeric_features = config['numeric_features']
        categorical_features = config['categorical_features']

        logger.info(f"Numeric features: {numeric_features}")
        logger.info(f"Categorical features: {categorical_features}")

        # Check if feature columns exist in the DataFrame
        missing_numeric_features = [col for col in numeric_features if col not in X.columns]
        missing_categorical_features = [col for col in categorical_features if col not in X.columns]
        
        if missing_numeric_features or missing_categorical_features:
            raise HTTPException(status_code=400, detail=f"Missing features in the data: {missing_numeric_features + missing_categorical_features}")
        
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
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        model_accuracy = accuracy_score(y_test, y_pred)

        if model_accuracy < 0.8:
            logger.warning(f"Model accuracy is below 80%: {model_accuracy}. The model was not saved.")
            return {
                "message": "Model accuracy is below 80%. The model was not saved.",
                "model_accuracy": model_accuracy
            }

        models_dir = 'models'
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        model_path = os.path.join(models_dir, f'credit_model_{model_name}.pkl')
        preprocessor_path = os.path.join(models_dir, f'preprocessor_{model_name}.pkl')
        
        joblib.dump(pipeline.named_steps['model'], model_path)
        joblib.dump(pipeline.named_steps['preprocessor'], preprocessor_path)
        
        num_features = preprocessor.transformers_[0][1].named_steps['scaler'].get_feature_names_out()
        cat_features = preprocessor.transformers_[1][1].get_feature_names_out()
        feature_names = num_features.tolist() + cat_features.tolist()

        model = pipeline.named_steps['model']
        weights = model.coef_.flatten()

        if len(feature_names) != len(weights):
            logger.error(f"Feature names and weights length mismatch: {len(feature_names)} vs {len(weights)}")
            raise ValueError(f"Feature names and weights length mismatch: {len(feature_names)} vs {len(weights)}")

        feature_weights = dict(zip(feature_names, weights))

        # Save model details to the database
        model_details = {
            "model_name": model_name,
            "model_path": model_path,
            "preprocessor_path": preprocessor_path,
            "model_accuracy": model_accuracy,
            "feature_weights": feature_weights
        }

        model_collection.update_one(
            {"model_name": model_name},
            {"$set": model_details},
            upsert=True
        )


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
async def predict(model_name: str, features: Features):
    try:
        # Ensure MODEL_PATH points to the correct directory where models are stored
        models_dir = 'C:\\Users\\Admin\\Downloads\\ai\\models'
        
        # Construct paths for the model and preprocessor based on model_name
        model_path = os.path.join(models_dir, f'credit_model_{model_name}.pkl')
        preprocessor_path = os.path.join(models_dir, f'preprocessor_{model_name}.pkl')

        # Log paths for debugging
        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Loading preprocessor from: {preprocessor_path}")

        # Check if the model and preprocessor files exist
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model file not found at path: {model_path}")
        if not os.path.exists(preprocessor_path):
            raise HTTPException(status_code=404, detail=f"Preprocessor file not found at path: {preprocessor_path}")

        # Load the model and preprocessor
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        
        # Log model and preprocessor loaded successfully
        logger.info("Model and preprocessor loaded successfully.")

        # Convert features to DataFrame
        feature_values = pd.DataFrame([features.dict()])

        # Log feature values for debugging
        logger.info(f"Features received: {feature_values}")

        # Process the features using the preprocessor
        processed_features = preprocessor.transform(feature_values)
        
        # Log processed features for debugging
        logger.info(f"Processed features: {processed_features}")

        # Make prediction
        prediction = model.predict(processed_features)
        prediction_proba = model.predict_proba(processed_features)[0][1]

        # Define creditworthiness based on prediction
        creditworthiness = "Good" if prediction[0] == 1 else "Bad"

        # Create response
        response = PredictionResponse(
            default_probability=prediction_proba,
            creditworthiness=creditworthiness,
            model_confidence=model.score(processed_features, prediction),
            ai_credit_score=prediction_proba * 100,
            recommendations=["Increase credit score", "Reduce debt"],
            model_accuracy=model.score(processed_features, prediction),
            feature_weights=dict(zip(preprocessor.get_feature_names_out(), model.coef_.flatten()))
        )
        
        

        logger.info(f"Prediction completed successfully: {response}")
        return response
    except Exception as e:
        # Log detailed error message and stack trace
        logger.error(f"Error making prediction: {str(e)}")
        logger.error("Stack trace: " + traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict12")
async def predict(model_name: str, features: Features):
    try:
        # Ensure MODEL_PATH points to the correct directory where models are stored
        models_dir = '/app/models'
        
        # Construct paths for the model and preprocessor based on model_name
        model_path = os.path.join(models_dir, f'credit_model_{model_name}.pkl')
        preprocessor_path = os.path.join(models_dir, f'preprocessor_{model_name}.pkl')

        # Log paths for debugging
        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Loading preprocessor from: {preprocessor_path}")

        # Check if the model and preprocessor files exist
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model file not found at path: {model_path}")
        if not os.path.exists(preprocessor_path):
            raise HTTPException(status_code=404, detail=f"Preprocessor file not found at path: {preprocessor_path}")

        # Load the model and preprocessor
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        
        # Log model and preprocessor loaded successfully
        logger.info("Model and preprocessor loaded successfully.")

        # Convert features to DataFrame
        feature_values = pd.DataFrame([features.dict()])

        # Log feature values for debugging
        logger.info(f"Features received: {feature_values}")

        # Process the features using the preprocessor
        processed_features = preprocessor.transform(feature_values)
        
        # Log processed features for debugging
        logger.info(f"Processed features: {processed_features}")

        # Make prediction
        prediction = model.predict(processed_features)
        prediction_proba = model.predict_proba(processed_features)[0][1]

        # Define creditworthiness based on prediction
        creditworthiness = "Good" if prediction[0] == 1 else "Bad"

        # Create response
        response = PredictionResponse(
            default_probability=prediction_proba,
            creditworthiness=creditworthiness,
            model_confidence=model.score(processed_features, prediction),
            ai_credit_score=prediction_proba * 100,
            recommendations=["Increase credit score", "Reduce debt"],
            model_accuracy=model.score(processed_features, prediction),
            feature_weights=dict(zip(preprocessor.get_feature_names_out(), model.coef_.flatten()))
        )
        
        logger.info(f"Prediction completed successfully: {response}")
        return response
    except Exception as e:
        # Log detailed error message and stack trace
        logger.error(f"Error making prediction: {str(e)}")
        logger.error("Stack trace: " + traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predictpercentage")
async def predict(model_name: str, features: Features):
    try:
        # Ensure MODEL_PATH points to the correct directory where models are stored
        models_dir = 'C:\\Users\\Admin\\Downloads\\ai\\models'
        
        # Construct paths for the model and preprocessor based on model_name
        model_path = os.path.join(models_dir, f'credit_model_{model_name}.pkl')
        preprocessor_path = os.path.join(models_dir, f'preprocessor_{model_name}.pkl')

        # Log paths for debugging
        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Loading preprocessor from: {preprocessor_path}")

        # Check if the model and preprocessor files exist
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model file not found at path: {model_path}")
        if not os.path.exists(preprocessor_path):
            raise HTTPException(status_code=404, detail=f"Preprocessor file not found at path: {preprocessor_path}")

        # Load the model and preprocessor
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        
        # Log model and preprocessor loaded successfully
        logger.info("Model and preprocessor loaded successfully.")

        # Convert features to DataFrame
        feature_values = pd.DataFrame([features.dict()])

        # Log feature values for debugging
        logger.info(f"Features received: {feature_values}")

        # Process the features using the preprocessor
        processed_features = preprocessor.transform(feature_values)
        
        # Log processed features for debugging
        logger.info(f"Processed features: {processed_features}")

        # Make prediction
        prediction = model.predict(processed_features)
        prediction_proba = model.predict_proba(processed_features)[0][1]

        # Define creditworthiness based on prediction
        creditworthiness = "Good" if prediction[0] == 1 else "Bad"

        # Create response
        response = PredictionResponse(
            default_probability=prediction_proba * 100,
            creditworthiness=creditworthiness,
            model_confidence=100.0,  # Since model.score returns 1 for full confidence
            ai_credit_score=prediction_proba * 100 * 100,
            recommendations=["Increase credit score", "Reduce debt"],
            model_accuracy=100.0,  # Since model.score returns 1 for full accuracy
            feature_weights=dict(zip(preprocessor.get_feature_names_out(), model.coef_.flatten()))
        )
        
        logger.info(f"Prediction completed successfully: {response}")
        return response
    except Exception as e:
        # Log detailed error message and stack trace
        logger.error(f"Error making prediction: {str(e)}")
        logger.error("Stack trace: " + traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict1save")
async def predict(model_name: str, features: Features):
    try:
        # Ensure MODEL_PATH points to the correct directory where models are stored
        models_dir = 'C:\\Users\\Admin\\Downloads\\ai\\models'
        
        # Construct paths for the model and preprocessor based on model_name
        model_path = os.path.join(models_dir, f'credit_model_{model_name}.pkl')
        preprocessor_path = os.path.join(models_dir, f'preprocessor_{model_name}.pkl')

        # Log paths for debugging
        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Loading preprocessor from: {preprocessor_path}")

        # Check if the model and preprocessor files exist
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model file not found at path: {model_path}")
        if not os.path.exists(preprocessor_path):
            raise HTTPException(status_code=404, detail=f"Preprocessor file not found at path: {preprocessor_path}")

        # Load the model and preprocessor
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        
        # Log model and preprocessor loaded successfully
        logger.info("Model and preprocessor loaded successfully.")

        # Convert features to DataFrame
        feature_values = pd.DataFrame([features.dict()])

        # Log feature values for debugging
        logger.info(f"Features received: {feature_values}")

        # Process the features using the preprocessor
        processed_features = preprocessor.transform(feature_values)
        
        # Log processed features for debugging
        logger.info(f"Processed features: {processed_features}")

        # Make prediction
        prediction = model.predict(processed_features)
        prediction_proba = model.predict_proba(processed_features)[0][1]

        # Define creditworthiness based on prediction
        creditworthiness = "Good" if prediction[0] == 1 else "Bad"

        # Calculate model confidence and accuracy
        # Note: model.score() requires both features and true labels; it's not suitable for a single prediction
        # Assuming model.score() is not applicable for single prediction, set it to 100 for this example
        model_confidence = 100.0
        model_accuracy = 100.0

        # Create response
        response = PredictionResponse(
            default_probability=prediction_proba * 100,  # Convert to percentage
            creditworthiness=creditworthiness,
            model_confidence=model_confidence,
            ai_credit_score=prediction_proba * 1000,  # Adjust as needed
            recommendations=["Increase credit score", "Reduce debt"],
            model_accuracy=model_accuracy,
            feature_weights=dict(zip(preprocessor.get_feature_names_out(), model.coef_.flatten()))
        )
        
        # Save prediction details to the database
        prediction_details = {
            "model_name": model_name,
            "features": features.dict(),
            "prediction": creditworthiness,
            "probability": prediction_proba * 100,  # Convert to percentage
            "ai_credit_score": prediction_proba * 1000,  # Adjust as needed
            "model_confidence": model_confidence,
            "model_accuracy": model_accuracy,
            "feature_weights": dict(zip(preprocessor.get_feature_names_out(), model.coef_.flatten()))
        }

        predictions_collection.insert_one(prediction_details)

        logger.info(f"Prediction completed successfully and saved to database: {response}")
        return response
    except Exception as e:
        # Log detailed error message and stack trace
        logger.error(f"Error making prediction: {str(e)}")
        logger.error("Stack trace: " + traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
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
