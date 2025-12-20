import os
import json
import tempfile
import boto3
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from schemas import PredictionResponse, DiseasePrediction, DiseasesResponse
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
from contextlib import asynccontextmanager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# récupération config par vars d'env
load_dotenv()
DEVICE='cpu'
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI","https://gviel-mlflow37.hf.space")
#structure MLFlow URI : s3://<bucket-name>/<mlflow_dir_name>/<experiment_id>/models/m-<model-uuid>/artifacts/data/model.pth
MLFLOW_MODEL_URI = os.getenv("MLFLOW_MODEL_URI", "s3://aws-s3-mlflow/mlflow-artifacts/3/models/m-46e598be60f940849247fc01cf53dc3c/artifacts/data/model.pth")
DATASET_NAME = os.getenv("DATASET_NAME", "kaggle")

def load_disease(bucket_name:str, dataset_name:str) -> dict:
    '''
    Chargement des maladies à partir du bucket S3.

    On récupère un fichier vitiscan-data/diseases-{DATASET_NAME}.json que l'on
    charge dans un dictionnaire global DISEASES={ 'disease1_name' : 'disease1_translated', ...}

    :param bucket_name: nom du bucket S3
    :type bucket_name: str
    :param dataset_name: nom du dataset
    :type file_path: str
    
    '''
    diseases = { 'N/A' : 'N/A' }
    file_path = f'vitiscan-data/diseases-{dataset_name}.json'
    try:
        response = S3_CLIENT.get_object(Bucket=bucket_name, Key=file_path)
        data = response['Body'].read().decode('utf-8')
        diseases = json.loads(data)
    except:
        logger.error(f'Impossible to retrieve diseases from s3://{bucket_name}/{file_path}')
        
    return diseases    

def load_model_from_s3(s3_bucket_name:str, s3_artifact_uri:str):
    '''
    Chargement du modèle à partir du bucket S3
    '''
    # calcul taille du fichier du modèle
    response = S3_CLIENT.head_object(Bucket=s3_bucket_name, Key=s3_artifact_uri)
    file_size = response['ContentLength']
    logger.info(f"model_file_size = {file_size}")

    # téléchargement vers un fichier temporaire
    with tempfile.NamedTemporaryFile(prefix="model", text=False) as tmp_file:
        model_local_path = tmp_file.name
        with tqdm(total=file_size, unit='B', unit_scale=True, desc='Téléchargement') as pbar:
            S3_CLIENT.download_file(
                s3_bucket_name,
                s3_artifact_uri,
                model_local_path,
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred)
        )
        logger.info(f"Model downloaded to {model_local_path}")

        # chargement du modèle en mémoire en forçant vers le CPU
        model = torch.load(model_local_path, map_location=DEVICE)
        logger.info(f"Model successfuly loaded from {model_local_path}")
        return model

def predict_image(model, input_tensor: torch.Tensor) -> list:
    '''
        Docstring pour predict_image
    
        :param model: Description
        :param input_tensor: Description
        :type input_tensor: torch.Tensor
    '''
    with torch.no_grad():
        model.eval()
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        predictions = [(DISEASES.keys()[i], float(probs[i])) for i in range(len(DISEASES))] # Création de la liste DiseasePrediction
        predictions.sort(key=lambda x: x[1], reverse=True) # Tri décroissante de la confiance
    return predictions

async def startup():
    '''
    Méthode de démarrage de l'API :
    - création bucket S3
    - récupération du model sur MLFlow
    - récupération du fichier de config des maladies
    - instanciation du tranform pour les images avant le predict
    '''
    # Code de configuration ici
    print("API startup and configuration")
    
    # declare global vars
    global S3_CLIENT
    global DISEASES
    #global EXPERIMENT
    #global EXPERIMENT_NAME
    global MODEL
    global MODEL_ID
    global MODEL_NAME
    global TRANSFORM
    
    try:
        # create s3 client
        S3_CLIENT = boto3.client('s3')

        # récupération des configs S3 à partir de l'URI du model
        mlflow_model_uri_splitted = MLFLOW_MODEL_URI.replace("s3://","").split("/")
        s3_bucket_name = mlflow_model_uri_splitted[0]
        s3_artifact_uri = MLFLOW_MODEL_URI.replace("s3://" + s3_bucket_name + "/", "")
        logger.info(f"s3_bucket_name={s3_bucket_name}")
        logger.info(f"s3_artifact_uri={s3_artifact_uri}")

        # chargement du modèle
        MODEL = load_model_from_s3(s3_bucket_name, s3_artifact_uri)

        # récupération de l'experiment_id et experiment_name
        #experiment_id = mlflow_model_uri_splitted[2]
        MODEL_ID = mlflow_model_uri_splitted[4]
        logger.info(f"MODEL_ID=%", MODEL_ID)
        logged_model = mlflow.get_logged_model(MODEL_ID)
        MODEL_NAME = logged_model.name
        logger.info(f"MODEL_NAME=%", MODEL_NAME)
        #EXPERIMENT = mlflow.get_experiment(experiment_id)
        #EXPERIMENT_NAME = EXPERIMENT.name
        #logger.info(f"EXPERIMENT_NAME=%", EXPERIMENT_NAME)

        # chargement des maladies correspondant au modèle
        try:
            # on essaye de charger à partir des extra-files du modèle
            disease_filename = 'diseases-{DATASET_NAME}.json'
            extra_files_dir = '/'.join(mlflow_model_uri_splitted[1:6]+['extra_files'])
            file_path = f'{extra_files_dir}/{disease_filename}'
            DISEASES = load_disease(bucket_name=s3_bucket_name, file_path=file_path)
        except:
            # sinon on va chercher à la base du bucket dans vitiscan-data
            file_path=f'vitiscan-data/diseases-{DATASET_NAME}.json'
            DISEASES = load_disease(bucket_name=s3_bucket_name, file_path=file_path)
        finally:
            logger.info("Disease dictionnary downloaded from :", file_path)
            logger.info(json.dumps(DISEASES, indent=4, ensure_ascii=True))

        # transformation à appliquer pour la prédiction (sans le random ColorJitter)
        TRANSFORM = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])

        # fermeture du client S3
        S3_CLIENT.close()
    except:
        logger.error("Error during init of API diagno")
    finally:
        S3_CLIENT.close()

async def shutdown():
    ''' Code de sortie de l'API '''
    logger.info("API shutdown")

@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup()  # Appel de la fonction de démarrage
    yield  # Indique que l'application est prête
    await shutdown()  # Appel de la fonction de nettoyage

app = FastAPI(lifespan=lifespan, title="VitiScan Diagno API")

@app.get("/")
def root():
    return {"message": "Vitiscan Diagno API is running"}

@app.get("/diseases", response_model=DiseasesResponse)
async def diseases():
    return DiseasesResponse(
        diseases=DISEASES,
        dataset_name=DATASET_NAME
    )

@app.post("/diagno", response_model=PredictionResponse)
async def diagno(file: UploadFile = File(...)):
    '''
    Diagnostic à partir d'un fichier image envoyé par l'interface web.
    
    :param file: Description
    :type file: UploadFile
    '''
    if file is None:
        return JSONResponse(status_code=400, content={"message": "Aucun fichier reçu"})
    
    # on devrait récupèrer l'image sous forme de bytes directement et créer l'image PIL mais ne fonctionne pas
    # contents = await file.read()
    #pil_image = Image.open(io.BytesIO(contents))

    # pour l'instant obligé de passer par un fichier temporaire
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        contents = await file.read()
        tmp_file.write(contents)
        tmp_file_path = tmp_file.name
        predictions = []
        try:
            # on lit l'image sous forme de tensor et 
            tensor_image = read_image(tmp_file_path)
            image = to_pil_image(tensor_image) # on fait ensuite la conversion en Image PIL
            img_converted = image.convert("RGB") # on convertit en RGB
            tensor = TRANSFORM(img_converted).unsqueeze(0)
            # on force le modèle et les datas en CPU
            device = next(MODEL.parameters()).device
            logger.debug(f'The model is on: {device}')
            tensor.to(DEVICE)
            MODEL.to(DEVICE)
            # Réalisation de la prédiction
            raw_predictions = predict_image(MODEL, tensor)
            predictions = [DiseasePrediction(disease=d, confidence=c) for d, c in raw_predictions]
            tmp_file.close()
        except:
            return JSONResponse(status_code=500, content={"message": "Predict error in API diagno"})
        finally:
            os.unlink(tmp_file_path)
    
    return PredictionResponse(
            predictions = predictions,
            model_version = MODEL_NAME
        )

if __name__ == "__main__":
    # method called only in local mode
    uvicorn.run("app:app", host="127.0.0.1", port=4000, reload=True)
