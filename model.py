'''
# import mlflow
# from PIL import Image
# import torch
# import torchvision.transforms as transforms
#
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