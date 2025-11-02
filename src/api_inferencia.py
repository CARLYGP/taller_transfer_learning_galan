from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import torch
from torchvision import transforms, models, datasets
import torch.nn as nn
import io

app = FastAPI(title="API de Inferencia CNN", version="1.0")

MODEL_NAME = "vgg16"          # "vgg16" o "resnet50"
NUM_CLASSES = 10              # 10 para CIFAR-10, 100 para CIFAR-100
DATASET_NAME = "cifar10"      # "cifar10" o "cifar100"
WEIGHTS_PATH = "modelos/vgg16_cifar10.pth"  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_model(model_name: str, num_classes: int, weights_path: str):
    """Carga un modelo (VGG16 o ResNet50) con los pesos entrenados."""
    if model_name.lower() == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        for param in model.features.parameters():
            param.requires_grad = False
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)

    elif model_name.lower() == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    else:
        raise ValueError("Modelo no soportado. Usa 'vgg16' o 'resnet50'.")

    # Cargar pesos del modelo entrenado
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Modelo '{model_name}' cargado desde: {weights_path}")
    return model

# Cargar modelo y etiquetas al iniciar el servidor
model = load_model(MODEL_NAME, NUM_CLASSES, WEIGHTS_PATH)

if DATASET_NAME.lower() == "cifar10":
    classes = datasets.CIFAR10(root="./data", train=False, download=True).classes
elif DATASET_NAME.lower() == "cifar100":
    classes = datasets.CIFAR100(root="./data", train=False, download=True).classes
else:
    raise ValueError("DATASET_NAME debe ser 'cifar10' o 'cifar100'.")


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "API de inferencia funcionando",
        "modelo": MODEL_NAME,
        "dataset": DATASET_NAME,
        "clases": len(classes)
    }

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Recibe una imagen y devuelve la clase predicha.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer la imagen: {e}")

    # Preprocesar la imagen
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Inferencia
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        label = classes[predicted.item()]

    return {"prediction": label}
