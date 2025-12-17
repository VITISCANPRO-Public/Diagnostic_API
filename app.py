import os
import io
import tempfile
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

DEVICE='cpu'
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI","https://gviel-mlflow37.hf.space")
#MLFLOW_MODEL_URI = os.getenv("MLFLOW_MODEL_URI", "mlflow-artifacts:/1/models/m-d4b6b9639a2b461b882af6c1ef3fbc61/artifacts") # modele GPU du 15/12 matin
MLFLOW_MODEL_URI = os.getenv("MLFLOW_MODEL_URI", "s3://aws-s3-mlflow/mlflow-artifacts/3/models/m-46e598be60f940849247fc01cf53dc3c/artifacts/data/model.pth") # modèle CPU du 15/12 soir
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MODEL = torch.load(MLFLOW_MODEL_URI, map_location="cpu")
MODEL.eval()
EXPERIMENT_NAME= os.getenv("EXPERIMENT_NAME")

#RUN_ID="2ac846b9752d4561ba7fa58864fec52a"
#run = mlflow.get_run(RUN_ID)
#experiment_id = run.info.experiment_id
#experiment = mlflow.get_experiment(experiment_id)
#EXPERIMENT_NAME = experiment.name
print("EXPERIMENT_NAME=", EXPERIMENT_NAME)


transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ]) 

app = FastAPI(title="VitiScan Diagno API")

@app.get("/")
def root():
    return {"message": "Vitiscan Diagno API is running"}

@app.post("/diagno", response_model=PredictionResponse)
async def diagno(file: UploadFile = File(...)):
    if file is None:
        return JSONResponse(status_code=400, content={"message": "Aucun fichier reçu"})
    
    # on devrait récupèrer l'image sous forme de bytes directement et créer l'image PIL
    # contents = await file.read()
    #pil_image = Image.open(io.BytesIO(contents))

    # pour l'instant obligé de passer par un fichier temporaire
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        contents = await file.read()
        tmp_file.write(contents)
        tmp_file_path = tmp_file.name

    try:
        # on lit l'image sous forme de tensor et 
        tensor_image = read_image(tmp_file_path)
        image = to_pil_image(tensor_image) # on fait ensuite la conversion en Image PIL
        img_converted = image.convert("RGB") # on convertit en RGB
        tensor = transform(img_converted).unsqueeze(0)
        # on force le modèle et les datas en CPU
        device = next(MODEL.parameters()).device
        print(f'DEBUG : The model is on: {device}')
        tensor.to(DEVICE)
        MODEL.to(DEVICE)
        raw_predictions = predict_image(MODEL, tensor) # Prédictions
        predictions = [DiseasePrediction(disease=d, confidence=c) for d, c in raw_predictions]
    except:
        predictions = []
    finally:
        os.unlink(tmp_file_path)  
    
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


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=4000, reload=True)
