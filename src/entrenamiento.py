import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("modelos", exist_ok=True)


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


# CARGA DE CIFAR-10 Y CIFAR-100
train_c10 = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
test_c10  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

train_c100 = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
test_c100  = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)


# DIVISIÓN TRAIN/VAL (80/20)
def split_dataset(dataset, val_split=0.2):
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    return random_split(dataset, [train_size, val_size])

train_c10, val_c10 = split_dataset(train_c10)
train_c100, val_c100 = split_dataset(train_c100)

def make_loaders(train_set, val_set, test_set, batch_size=64):
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2),
        DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    )

train_loader_c10, val_loader_c10, test_loader_c10 = make_loaders(train_c10, val_c10, test_c10)
train_loader_c100, val_loader_c100, test_loader_c100 = make_loaders(train_c100, val_c100, test_c100)


def train_model(model, train_loader, val_loader, epochs=7, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    history = []

    for epoch in range(epochs):
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_loss /= total_train
        train_acc = 100 * correct_train / total_train

        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= total_val
        val_acc = 100 * correct_val / total_val
        val_f1 = f1_score(all_labels, all_preds, average='macro')

        history.append({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1
        })

        print(f"Epoch {epoch+1}: TrainLoss={train_loss:.4f}, ValLoss={val_loss:.4f}, "
              f"TrainAcc={train_acc:.2f}%, ValAcc={val_acc:.2f}%, ValF1={val_f1:.4f}")

    print("Entrenamiento finalizado ")
    return history


# ENTRENAMIENTO VGG16 (CIFAR-10)
print("\n Entrenando VGG16 en CIFAR-10")
vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
for p in vgg16.features.parameters():
    p.requires_grad = False

num_features = vgg16.classifier[6].in_features
vgg16.classifier[6] = nn.Linear(num_features, 10)
vgg16 = vgg16.to(device)

history_vgg = train_model(vgg16, train_loader_c10, val_loader_c10)
torch.save(vgg16.state_dict(), "modelos/vgg16_cifar10.pth")
print("Guardado: modelos/vgg16_cifar10.pth")

# ENTRENAMIENTO RESNET50 (CIFAR-100)

print("\n Entrenando ResNet50 en CIFAR-100")
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
for p in resnet50.parameters():
    p.requires_grad = False
resnet50.fc = nn.Linear(resnet50.fc.in_features, 100)
for p in resnet50.fc.parameters():
    p.requires_grad = True
resnet50 = resnet50.to(device)

history_resnet = train_model(resnet50, train_loader_c100, val_loader_c100)
torch.save(resnet50.state_dict(), "modelos/resnet50_cifar100.pth")
print("Guardado: modelos/resnet50_cifar100.pth")


# MOSTRAR MÉTRICAS FINALES
print("\n Métricas finales VGG16:")
print(history_vgg[-1])
print("\n Métricas finales ResNet50:")
print(history_resnet[-1])


