import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from schemas import PredictionResponse, DiseasePrediction
#from model import preprocess_image, predict_image
import uvicorn
import mlflow
from PIL import Image
import torch
import torchvision.transforms as transforms
#from torchvision.transforms import v2
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI","https://gviel-mlflow37.hf.space")
MLFLOW_MODEL_URI = os.getenv("MLFLOW_MODEL_URI", "mlflow-artifacts:/1/models/m-d4b6b9639a2b461b882af6c1ef3fbc61/artifacts")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MODEL = mlflow.pytorch.load_model(MLFLOW_MODEL_URI)
MODEL.eval()

RUN_ID="2ac846b9752d4561ba7fa58864fec52a"
run = mlflow.get_run(RUN_ID)
experiment_id = run.info.experiment_id
experiment = mlflow.get_experiment(experiment_id)
EXPERIMENT_NAME = experiment.name
print("EXPERIMENT_NAME=", EXPERIMENT_NAME)


transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ]) 

app = FastAPI(title="VitiScan Diagno API")

@app.get("/")
def root():
    return {"message": "Vitiscan Diagno API is running"}

'''
# Réponse factice
@app.post("/diagno", response_model=PredictionResponse)
async def diagno(file: UploadFile = File(...)):
    if file is None:
        return JSONResponse(status_code=400, content={"message": "Aucun fichier reçu"})
    
    # Réponse factice : mock response
    # fake_prediction = "mildiou"
    # fake_confidence = 0.87
    # fake_model_version = "Resnet18_50ep_v2"

    predictions = [DiseasePrediction(disease="mildiou", confidence=0.87)]
    return PredictionResponse(
        predictions=predictions,
        model_version="Resnet18_50ep_v2"
    )

    # return PredictionResponse(
    #     disease=fake_prediction,
    #     confidence=fake_confidence,
    #     model_version=fake_model_version
    # )
'''

@app.post("/diagno", response_model=PredictionResponse)
async def diagno(file: UploadFile = File(...)):
    if file is None:
        return JSONResponse(status_code=400, content={"message": "Aucun fichier reçu"})
    
    #contents = await file.read()
    #print("filename=", file.filename)
    #print("type(filename)=", type(file.filename))
    tensor_image = read_image(file.filename)
    image = to_pil_image(tensor_image)
    img_converted = image.convert("RGB")
    tensor = transform(img_converted).unsqueeze(0)
    device = next(MODEL.parameters()).device
    print(f'The model is on: {device}')
    tensor.to("cpu")
    MODEL.to("cpu")
    raw_predictions = predict_image(MODEL, tensor) # Prédictions
    predictions = [DiseasePrediction(disease=d, confidence=c) for d, c in raw_predictions]
    
    return PredictionResponse(
            predictions = predictions,
            model_version = EXPERIMENT_NAME
        )

# ---------------------------------  PREDICTION -------------------------------
# Classes names
disease = [
    "anthracnose",
    "brown_spot",
    "downy_mildew",
    "mites",
    "normal",
    "powdery_mildew",
    "shot_hole"
]

def predict_image(model, input_tensor: torch.Tensor):
    print(type(input_tensor))
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        predictions = [(disease[i], float(probs[i])) for i in range(len(disease))] # Création de la liste DiseasePrediction
        predictions.sort(key=lambda x: x[1], reverse=True) # Tri décroissante de la confiance
    return predictions

'''
def preprocess_image(img_filename:str) -> torch.Tensor:
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ]) 
    #image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    #with open(img_filename, 'rb') as f:
    print(type(img_filename))
    image = Image.open(img_filename) # return ImageFile
    img_converted = image.convert("RGB") # renvoie PIL.Image.Image
    #to_pil = transforms.ToPILImage()
    #img_from_tensor = to_pil(img_converted)
    tensor = transform(img_converted).unsqueeze(0)
    return tensor
'''

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=4000, reload=True)
