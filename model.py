import mlflow
from PIL import Image
import torch
import torchvision.transforms as transforms

disease = [
    "anthracnose",
    "brown_spot",
    "downy_mildew",
    "mites",
    "normal",
    "powdery_mildew",
    "shot_hole"
]

# Chargement depuis URI direct (ex:s3)
#def load_model():
#    model_uri = "s3://aws-s3-mlflow/mlflow-artifacts/"
#    model = mlflow.pytorch.load_model(model_uri)
#    model.eval()   
#    return model

# Chargement depuis MLFlow (nom du modèle)
def load_model(mlflow_uri: str):
    mlflow.set_tracking_uri(mlflow_uri)
    model = mlflow.pytorch.load_model("models:/grape_disease/latest") # remplacer par le nom du modèle
    model.eval()
    return model

def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict_image(model, image: Image.Image):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0] 
        predictions = [(disease[i], float(probs[i])) for i in range(len(disease))] # Création de la liste DiseasePrediction
        predictions.sort(key=lambda x: x[1], reverse=True) # Tri décroissante de la confiance
        #predictions = [
        #    DiseasePrediction(disease=disease[i], confidence=float(probs[i]))
        #    for i in range(len(disease))
        #]
        #predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions

'''
# def load_model(mlflow_uri: str):
#     mlflow.set_tracking_uri(mlflow_uri)
#     # Exemple : charger le dernier modèle enregistré
#     model = mlflow.pytorch.load_model("models:/grape_disease/latest")
#     model.eval()
#     return model
#
# def preprocess_image(image: Image.Image):
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])
#     return transform(image).unsqueeze(0)
#
# def predict_image(model, image: Image.Image):
#     input_tensor = preprocess_image(image)
#     with torch.no_grad():
#         output = model(input_tensor)
#         confidence, predicted_class = torch.max(torch.nn.functional.softmax(output, dim=1), 1)
#     disease = predicted_class.item()
#     return f"Disease_{disease}", float(confidence)
'''