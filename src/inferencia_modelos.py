import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

device = torch.device("cpu")
print("Usando CPU para inferencia")

TEST_PATHS = {
    "vgg16": "test/VGG16",
    "resnet50": "test/Resnet"
}

MODELS_PATHS = {
    "vgg16": "modelos/vgg16_cifar10.pth",
    "resnet50": "modelos/resnet50_cifar100.pth"
}

CONFIDENCE_THRESHOLD = 0.6

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_model(model_name: str, num_classes: int, weights_path: str):
    if model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        for param in model.features.parameters():
            param.requires_grad = False
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError("Modelo no soportado")

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Modelo {model_name.upper()} cargado desde: {weights_path}")
    return model

def predict_image(image_path, model, class_names):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        max_prob, predicted = torch.max(probs, 1)
    confidence = float(max_prob.item())
    label = "unknown" if confidence < CONFIDENCE_THRESHOLD else class_names[predicted.item()]
    return label, confidence

def evaluate_model(model_name, model, test_folder, dataset_name):
    print(f"\nEvaluando modelo {model_name.upper()} en {test_folder}")
    if dataset_name == "cifar10":
        dataset = datasets.CIFAR10(root="./data", train=False, download=True)
    else:
        dataset = datasets.CIFAR100(root="./data", train=False, download=True)
    class_names = dataset.classes
    print(f"{model_name.upper()} detecta {len(class_names)} clases: {class_names[:10]}{'...' if len(class_names) > 10 else ''}")

    results = []
    for img_file in tqdm(os.listdir(test_folder)):
        img_path = os.path.join(test_folder, img_file)
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        pred_label, conf = predict_image(img_path, model, class_names)
        results.append({
            "imagen": img_file,
            "prediccion": pred_label,
            "confianza": round(conf, 4),
            "above_threshold": conf >= CONFIDENCE_THRESHOLD
        })

    df = pd.DataFrame(results)
    csv_name = f"resultados_{model_name}.csv"
    df.to_csv(csv_name, index=False)
    print(f"Resultados guardados en {csv_name}")
    return df

if __name__ == "__main__":
    vgg = load_model("vgg16", 10, MODELS_PATHS["vgg16"])
    resnet = load_model("resnet50", 100, MODELS_PATHS["resnet50"])

    df_vgg = evaluate_model("vgg16", vgg, TEST_PATHS["vgg16"], "cifar10")
    df_res = evaluate_model("resnet50", resnet, TEST_PATHS["resnet50"], "cifar100")

    def resumen(df, nombre):
        total = len(df)
        unknowns = len(df[df["prediccion"] == "unknown"])
        conf_prom = df["confianza"].mean()
        return {"modelo": nombre, "total": total, "unknowns": unknowns, 
                "unknowns_%": (unknowns/total)*100 if total > 0 else 0, 
                "confianza_promedio": conf_prom}

    resumen_vgg = resumen(df_vgg, "VGG16 (10 clases)")
    resumen_res = resumen(df_res, "ResNet50 (100 clases)")

    print("\nResumen numérico:")
    for r in [resumen_vgg, resumen_res]:
        print(r)

    def mostrar_comparacion(df_vgg, folder_vgg, df_res, folder_res):
        n = min(5, len(df_vgg), len(df_res))
        sample_vgg = df_vgg.sample(n)
        sample_res = df_res.sample(n)
        fig, axes = plt.subplots(2, n, figsize=(15, 6))
        for i, row in enumerate(sample_vgg.itertuples(), 0):
            img_path = os.path.join(folder_vgg, row.imagen)
            img = Image.open(img_path).convert("RGB")
            axes[0, i].imshow(img)
            axes[0, i].axis("off")
            axes[0, i].set_title(f"{row.prediccion}\n{row.confianza:.2f}", fontsize=9)
        for i, row in enumerate(sample_res.itertuples(), 0):
            img_path = os.path.join(folder_res, row.imagen)
            img = Image.open(img_path).convert("RGB")
            axes[1, i].imshow(img)
            axes[1, i].axis("off")
            axes[1, i].set_title(f"{row.prediccion}\n{row.confianza:.2f}", fontsize=9)
        fig.suptitle("Fila 1: VGG16 (10 clases)   |   Fila 2: ResNet50 (100 clases)", fontsize=13, y=0.98)
        plt.tight_layout()
        plt.show()

    mostrar_comparacion(df_vgg, TEST_PATHS["vgg16"], df_res, TEST_PATHS["resnet50"])

    def mostrar_tabla_comparacion(resumen_vgg, resumen_res):
        fig, ax = plt.subplots(figsize=(6, 2))
        df = pd.DataFrame([resumen_vgg, resumen_res])
        ax.axis("off")
        tabla = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
        tabla.auto_set_font_size(False)
        tabla.set_fontsize(9)
        tabla.scale(1.1, 1.5)
        ax.set_title("Comparación de Confianza Promedio y Unknowns entre Modelos", fontsize=11, pad=10)
        plt.show()

    mostrar_tabla_comparacion(resumen_vgg, resumen_res)
