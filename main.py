# Import FastAPI class
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import xgboost as xgb
import pickle
from fastapi.middleware.cors import CORSMiddleware
from torchvision import models, transforms
from PIL import Image
import torch
import os
from fastapi import File, UploadFile
import io
from logger import log_diabetes_prediction  # <-- new function we’ll create


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],             # Allowed frontend domains
    allow_credentials=True,
    allow_methods=["*"],               # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"]                # Allow all headers (especially Content-Type)
)


# logging 
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="diabetic.log",  
    filemode="a"         
)

# 

class InputData(BaseModel):
    pregnancies: float
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree_function: float
    age: float

class HeartInput(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int



invalid_zero_features = [
    "glucose",         
    "blood_pressure", 
    "skin_thickness", 
    "insulin",        
]

invalid_zero_median = {
    "glucose":117.0,         
    "blood_pressure":72.0, 
    "skin_thickness":29.0, 
    "insulin":125.0,        
}

# loading the models
diabet_model = xgb.Booster()
diabet_model.load_model("./model/diabet-xgboost-model")
heart_model = xgb.Booster()
heart_model.load_model("./model/heart-xgboost-model")



@app.get("/")
def read_root():
    return {"message": "Dr. Evangadi API is working!"}


# ----------------------------

# Internal function (not FastAPI route!)
def preprocess_input(user_input: list[float]) -> list[float]:
    """
    Takes raw input data from user (8 features), replaces invalid zeros,
    adds invalid-zero flags, and returns final list with 12 features.
    """
    features = [
        "pregnancies", "glucose", "blood_pressure", "skin_thickness",
        "insulin", "bmi", "diabetes_pedigree_function", "age"
    ]
    
    # Invalid zero features and their medians
    invalid_zero_median = {
        "glucose": 117.0,
        "blood_pressure": 72.0,
        "skin_thickness": 29.0,
        "insulin": 125.0
    }

    processed = []
    invalid_flags = []

    for i, value in enumerate(user_input):
        feature = features[i]

        if feature in invalid_zero_median:
            if value == 0:
                processed.append(invalid_zero_median[feature])
                invalid_flags.append(1)
            else:
                processed.append(value)
                invalid_flags.append(0)
        else:
            processed.append(value)

    return processed + invalid_flags
# ------------------------------------------------------------
from fastapi import HTTPException

@app.post("/diabet")
def predict(data: InputData): 
    try:
        user_input = [
            data.pregnancies, data.glucose, data.blood_pressure,
            data.skin_thickness, data.insulin, data.bmi,
            data.diabetes_pedigree_function, data.age
        ]

        logging.info(f"Received input: {user_input}")
        
        final_input = preprocess_input(user_input)
        logging.info(f"Preprocessed input: {final_input}")

        final_array = xgb.DMatrix([final_input])
        pred_prob = diabet_model.predict(final_array)[0]
        label = int(pred_prob >= 0.5)

        logging.info(f"Prediction: label={label}, probability={pred_prob}")
        try:
            # Prepare dict for input (to save raw inputs with keys)
            input_dict = {
                "pregnancies": data.pregnancies,
                "glucose": data.glucose,
                "blood_pressure": data.blood_pressure,
                "skin_thickness": data.skin_thickness,
                "insulin": data.insulin,
                "bmi": data.bmi,
                "diabetes_pedigree_function": data.diabetes_pedigree_function,
                "age": data.age
            }
            log_diabetes_prediction(
                input_data=input_dict,
                prediction=label,
                probability=round(float(pred_prob), 2)
            )
        except Exception as log_err:
            logging.error(f"DynamoDB logging failed: {log_err}")

        return {
            "label": label,
            "probability": round(float(pred_prob), 2)
        }
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid input or prediction error.")
       

@app.post("/heart")
def predict(data: HeartInput):
    try:
        input_features = [
            data.age, data.sex, data.cp, data.trestbps, data.chol, data.fbs,
            data.restecg, data.thalach, data.exang, data.oldpeak, data.slope,
            data.ca, data.thal
        ]
        logging.info(f"Received input: {input_features}")

        # Convert input to DMatrix for XGBoost
        dmatrix = xgb.DMatrix([input_features])
        pred_prob = heart_model.predict(dmatrix)[0]
        label = int(pred_prob >= 0.5)

        logging.info(f"Prediction: label={label}, probability={pred_prob}")

        return {
            "label": label,
            "probability": round(float(pred_prob), 2)
        }

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail="Invalid input or prediction error.")
    

    # Load ResNet18 model for cancer prediction
cancer_model = models.resnet18(pretrained=False)
cancer_model.fc = torch.nn.Linear(cancer_model.fc.in_features, 3)
cancer_model.load_state_dict(torch.load("./model/model.pth", map_location=torch.device("cpu")))
cancer_model.eval()

# Class labels
cancer_classes = ['Adenocarcinoma', 'Squamous Cell Carcinoma', 'Normal']

# Transformation pipeline
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


@app.post("/cancer-predict")
async def predict_cancer_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = image_transform(image).unsqueeze(0)

        with torch.no_grad():
            output = cancer_model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            normal_prob = probabilities[2].item()  # 'Normal' class index
            threshold = 0.05  # Tune this threshold as needed
            
            if normal_prob > threshold:
                predicted_class = 2  # Predict 'Normal'
            else:
                predicted_class = torch.argmax(probabilities).item()

        return {
            "label": cancer_classes[predicted_class],
            "probabilities": [round(p.item(), 3) for p in probabilities]
        }

    except Exception as e:
        logging.error(f"Image prediction error: {e}")
        raise HTTPException(status_code=400, detail="Failed to process image or make prediction.")
