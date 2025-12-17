
# -----------------------  PREPROCESSING OF THE IMAGE ------------------------
def preprocess_image(img_filename:str) -> torch.Tensor:
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
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

#def predict_image(model, image: Image.Image):
def predict_image(model, input_tensor: torch.Tensor):
    #input_tensor = preprocess_image(image)
    print(type(input_tensor))
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        predictions = [(disease[i], float(probs[i])) for i in range(len(disease))] # Création de la liste DiseasePrediction
        predictions.sort(key=lambda x: x[1], reverse=True) # Tri décroissante de la confiance
    return predictions