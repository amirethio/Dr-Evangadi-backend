from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torchvision import models, transforms
from PIL import Image
import torch
import io
import logging
import boto3
from datetime import datetime
from decimal import Decimal
import uuid
import xgboost as xgb

# Initialize FastAPI
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="app.log",
    filemode="a"
)

# Setup DynamoDB
dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
table = dynamodb.Table("Predictions")

# ========== UTILS ==========
def convert_floats_to_decimal(data):
    if isinstance(data, list):
        return [convert_floats_to_decimal(i) for i in data]
    elif isinstance(data, dict):
        return {k: convert_floats_to_decimal(v) for k, v in data.items()}
    elif isinstance(data, float):
        return Decimal(str(data))
    else:
        return data

# ========== CANCER MODEL ==========
cancer_model = models.resnet18(pretrained=False)
cancer_model.fc = torch.nn.Linear(cancer_model.fc.in_features, 3)
cancer_model.load_state_dict(torch.load("./model/model.pth", map_location=torch.device("cpu")))
cancer_model.eval()

cancer_classes = ['Adenocarcinoma', 'Squamous Cell Carcinoma', 'Normal']
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def log_prediction(prediction_type: str, input_data: dict, prediction: any, confidence: any, extra: dict = None):
    item = {
        "id": str(uuid.uuid4()),
        "prediction_type": prediction_type,
        "prediction_result": prediction,
        "confidence": Decimal(str(round(confidence, 2))) if isinstance(confidence, float) else confidence,
        "timestamp": datetime.utcnow().isoformat(),
        "input_data": convert_floats_to_decimal(input_data),
    }
    if extra:
        for k, v in extra.items():
            item[k] = convert_floats_to_decimal(v)
    table.put_item(Item=item)

@app.post("/cancer-predict")
async def predict_cancer_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = image_transform(image).unsqueeze(0)

        with torch.no_grad():
            output = cancer_model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            normal_prob = probabilities[2].item()
            threshold = 0.05
            predicted_class = 2 if normal_prob > threshold else torch.argmax(probabilities).item()

        prob_list = [round(p.item(), 3) for p in probabilities]
        logging.info(f"Cancer prediction: label={cancer_classes[predicted_class]}, probabilities={prob_list}")

        try:
            log_prediction(
                prediction_type="cancer",
                input_data={"filename": file.filename},
                prediction=cancer_classes[predicted_class],
                confidence=normal_prob,
                extra={"probabilities": prob_list}
            )
        except Exception as e:
            logging.error(f"DynamoDB logging failed (cancer): {e}")

        return {
            "label": cancer_classes[predicted_class],
            "probabilities": prob_list
        }

    except Exception as e:
        logging.error(f"Cancer prediction error: {e}")
        raise HTTPException(status_code=400, detail="Failed to process image or make prediction.")

# ========== DIABETES MODEL ==========
class InputData(BaseModel):
    pregnancies: float
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree_function: float
    age: float

invalid_zero_median = {
    "glucose": 117.0,
    "blood_pressure": 72.0,
    "skin_thickness": 29.0,
    "insulin": 125.0,
}

diabet_model = xgb.Booster()
diabet_model.load_model("./model/diabet-xgboost-model")

def preprocess_input(user_input: list[float]) -> list[float]:
    features = [
        "pregnancies", "glucose", "blood_pressure", "skin_thickness",
        "insulin", "bmi", "diabetes_pedigree_function", "age"
    ]
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

def log_diabetes_prediction(input_data: dict, prediction: int, probability: float):
    item = {
        "id": str(uuid.uuid4()),
        "prediction_type": "diabetes",
        "prediction_result": prediction,
        "confidence": Decimal(str(round(probability, 2))),
        "timestamp": datetime.utcnow().isoformat(),
        "input_data": convert_floats_to_decimal(input_data),
    }
    table.put_item(Item=item)

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
            input_dict = data.dict()
            log_diabetes_prediction(
                input_data=input_dict,
                prediction=label,
                probability=pred_prob
            )
        except Exception as e:
            logging.error(f"DynamoDB logging failed: {e}")

        return {
            "label": label,
            "probability": round(float(pred_prob), 2)
        }

    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid input or prediction error.")

# ========== HEART MODEL ==========
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

heart_model = xgb.Booster()
heart_model.load_model("./model/heart-xgboost-model")

def log_heart_prediction(input_data: dict, prediction: int, probability: float):
    item = {
        "id": str(uuid.uuid4()),
        "prediction_type": "heart",
        "prediction_result": prediction,
        "confidence": Decimal(str(round(probability, 2))),
        "timestamp": datetime.utcnow().isoformat(),
        "input_data": convert_floats_to_decimal(input_data),
    }
    table.put_item(Item=item)

@app.post("/heart")
def predict_heart(data: HeartInput):
    try:
        input_features = [
            data.age, data.sex, data.cp, data.trestbps, data.chol, data.fbs,
            data.restecg, data.thalach, data.exang, data.oldpeak, data.slope,
            data.ca, data.thal
        ]
        logging.info(f"Received heart input: {input_features}")

        dmatrix = xgb.DMatrix([input_features])
        pred_prob = heart_model.predict(dmatrix)[0]
        label = int(pred_prob >= 0.5)
        logging.info(f"Heart prediction: label={label}, probability={pred_prob}")

        try:
            log_heart_prediction(
                input_data=data.dict(),
                prediction=label,
                probability=pred_prob
            )
        except Exception as e:
            logging.error(f"DynamoDB logging failed (heart): {e}")

        return {
            "label": label,
            "probability": round(float(pred_prob), 2)
        }

    except Exception as e:
        logging.error(f"Heart prediction error: {e}")
        raise HTTPException(status_code=400, detail="Invalid input or prediction error.")

# ========== ROOT ROUTE ==========
@app.get("/")
def root():
    return {"message": "API is working for cancer, diabetes, and heart disease predictions."}
