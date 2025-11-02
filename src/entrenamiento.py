
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("modelos", exist_ok=True)

# Transforms estándar de ImageNet
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# Datasets

train_c10 = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
test_c10  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

train_c100 = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
test_c100  = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)

train_loader_c10 = DataLoader(train_c10, batch_size=64, shuffle=True, num_workers=2)
test_loader_c10  = DataLoader(test_c10, batch_size=64, shuffle=False, num_workers=2)

train_loader_c100 = DataLoader(train_c100, batch_size=64, shuffle=True, num_workers=2)
test_loader_c100  = DataLoader(test_c100, batch_size=64, shuffle=False, num_workers=2)


def train_model(model, train_loader, test_loader, epochs=10, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    history = []  # para guardar métricas por época

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # Evaluación en test
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = 100 * (torch.tensor(all_preds) == torch.tensor(all_labels)).sum().item() / len(all_labels)
        f1 = f1_score(all_labels, all_preds, average='macro')

        history.append({"epoch": epoch+1, "loss": train_loss, "accuracy": acc, "f1_score": f1})
        print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Accuracy={acc:.2f}%, F1={f1:.4f}")

    # Métricas finales
    final_acc = history[-1]["accuracy"]
    final_f1 = history[-1]["f1_score"]
    print(" Métricas finales del modelo:")
    print(f"Accuracy final: {final_acc:.2f}%")
    print(f"F1-score final: {final_f1:.4f}")

    return history

# Entrenar VGG16 (CIFAR-10)
print("Entrenando VGG16 en CIFAR-10")
vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
for param in vgg16.features.parameters():
    param.requires_grad = False  # Congelar extractor

num_features = vgg16.classifier[6].in_features
vgg16.classifier[6] = nn.Linear(num_features, 10)  # 10 clases CIFAR-10
vgg16 = vgg16.to(device)

history_vgg = train_model(vgg16, train_loader_c10, test_loader_c10, epochs=10)
torch.save(vgg16.state_dict(), "modelos/vgg16_cifar10.pth")
print(" Guardado: modelos/vgg16_cifar10.pth")

# Entrenar ResNet50 (CIFAR-100)
print("Entrenando ResNet50 en CIFAR-100")
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
for param in resnet50.parameters():
    param.requires_grad = False  # Congelar todo
resnet50.fc = nn.Linear(resnet50.fc.in_features, 100)  # 100 clases CIFAR-100
for param in resnet50.fc.parameters():
    param.requires_grad = True  # Solo FC entrenable
resnet50 = resnet50.to(device)

history_resnet = train_model(resnet50, train_loader_c100, test_loader_c100, epochs=10)
torch.save(resnet50.state_dict(), "modelos/resnet50_cifar100.pth")
print("Guardado: modelos/resnet50_cifar100.pth")

print("Entrenamiento completado para ambos modelos.")

