import os
import io
import json
import tempfile
import boto3
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
import logging
import tempfile
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

DEVICE='cpu'
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI","https://gviel-mlflow37.hf.space")
#structure MLFlow URI : s3://<bucket-name>/<mlflow_dir_name>/<experiment_id>/models/m-<model-uuid>/artifacts/data/model.pth
MLFLOW_MODEL_URI = os.getenv("MLFLOW_MODEL_URI", "s3://aws-s3-mlflow/mlflow-artifacts/3/models/m-46e598be60f940849247fc01cf53dc3c/artifacts/data/model.pth")
DATASET_NAME = os.getenv("DATASET_NAME", "kaggle")

##########################
# chargement du modèle
##########################
#mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# préparation du download du S3
s3 = boto3.client('s3')
model_local_path = '/tmp/model.pth'
mlflow_model_uri_splitted = MLFLOW_MODEL_URI.replace("s3://","").split("/")
s3_bucket_name = mlflow_model_uri_splitted[0]
s3_artifact_uri = MLFLOW_MODEL_URI.replace("s3://"+s3_bucket_name+"/", "")
logger.info(f"s3_bucket_name={s3_bucket_name}")
logger.info(f"s3_artifact_uri={s3_artifact_uri}")
logger.info(f"model_local_path={model_local_path}")

# calcul taille du fichier du modèle
response = s3.head_object(Bucket=s3_bucket_name, Key=s3_artifact_uri)
file_size = response['ContentLength']
logger.info(f"model_file_size={file_size}")

# récupération de l'experiment_id et experiment_name
experiment_id = mlflow_model_uri_splitted[2]
experiment = mlflow.get_experiment(experiment_id)
EXPERIMENT_NAME = experiment.name
#logger.info(f"EXPERIMENT_NAME={EXPERIMENT_NAME}")
logger.info("EXPERIMENT_NAME=%s", EXPERIMENT_NAME)

with tqdm(total=file_size, unit='B', unit_scale=True, desc='Téléchargement') as pbar:
    s3.download_file(
        s3_bucket_name,
        s3_artifact_uri,
        model_local_path,
        Callback=lambda bytes_transferred: pbar.update(bytes_transferred)
)

# chargement du modèle en mémoire en forçant vers le CPU
logger.info(f"model downloaded !")
MODEL = torch.load(model_local_path, map_location=DEVICE)
MODEL.eval()

# chargement des maladies
load_disease(bucket_name=s3_bucket_name, file_path=f'vitiscan-data/diseases-{DATASET_NAME}.json')

# transformation à appliquer pour la prédiction (sans le random ColorJitter)
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
        # Réalisation de la prédiction
        raw_predictions = predict_image(MODEL, tensor)
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
'''
DISEASES = [
    "anthracnose",
    "brown_spot",
    "downy_mildew",
    "mites",
    "normal",
    "powdery_mildew",
    "shot_hole"
]
'''
def load_disease(bucket_name:str, file_path:str):
    # chargement des maladies
    global DISEASES
    try:
        response = s3.get_object(Bucket=bucket_name, Key=file_path)
        data = response['Body'].read().decode('utf-8')
        DISEASES = json.loads(data)
    except:
        DISEASES = { 'N/A' : 'N/A' }

    with tempfile.TemporaryDirectory() as tempdir:
        disease_filename = f'disease-{DATASET_NAME}.json'
        tmpfile = str(Path(tempdir, disease_filename))
        s3.download_file(
            s3_bucket_name,
            f'vitiscan-data/{disease_filename}',
            tmpfile
        )
        os.unlink(tempdir)
        

def predict_image(model, input_tensor: torch.Tensor):
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        predictions = [(DISEASES[i], float(probs[i])) for i in range(len(DISEASES))] # Création de la liste DiseasePrediction
        predictions.sort(key=lambda x: x[1], reverse=True) # Tri décroissante de la confiance
    return predictions


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=4000, reload=True)
