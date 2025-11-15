import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os
import cv2
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################### from here
# 이미지의 채도(Saturation) 낮추는 함수
def reduce_saturation(image, saturation_factor):
    """
    이미지에서 채도를 낮추는 함수
    saturation_factor: 0.0 ~ 1.0, 0이면 완전히 흑백, 1이면 원래 상태
    """
    # 이미지를 numpy array로 변환
    image_np = np.array(image)

    # BGR로 변환 (OpenCV는 BGR을 사용)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # HSV로 변환
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Saturation 값 조정
    image_hsv[..., 1] = image_hsv[..., 1] * saturation_factor

    # Saturation이 조정된 이미지를 다시 BGR로 변환
    image_bgr_saturation_reduced = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    # 결과를 다시 RGB로 변환
    result_rgb = cv2.cvtColor(image_bgr_saturation_reduced, cv2.COLOR_BGR2RGB)

    return result_rgb

# 채도 낮추는 transform
class SaturationTransform:
    def __init__(self, saturation_factor):
        self.saturation_factor = saturation_factor

    def __call__(self, image):
        return reduce_saturation(image, self.saturation_factor)
########################### to here

def compute_mean_std(dataset_root):
    data_transforms = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root=dataset_root, transform=data_transforms)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    mean = 0.0
    std = 0.0
    total_images = 0

    for inputs, _ in loader:
        total_images += inputs.size(0)
        mean += inputs.mean([0, 2, 3]) * inputs.size(0)
        std += inputs.std([0, 2, 3]) * inputs.size(0)

    mean /= total_images
    std /= total_images

    return mean.tolist(), std.tolist()

train_mean, train_std = compute_mean_std('training')

########################### from here
# 채도 낮추기
saturation_factor = 0.3  # 채도를 30%로 줄임 (0은 완전히 흑백, 1은 원래 상태)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    SaturationTransform(saturation_factor=0.3),  # 채도를 30%로 낮춤
    transforms.ToTensor(),
    transforms.Normalize(mean=train_mean, std=train_std),
])
########################### to here

train_dataset = datasets.ImageFolder(root='augmented_training', transform=transform)
val_dataset = datasets.ImageFolder(root='augmented_validation', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(train_dataset.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

num_epochs = 10
batch_size = 32
learning_rate = 0.0001
optimizer_type = "AdamW"
loss_function = "CrossEntropyLoss"

train_losses = []
val_losses = []
accuracies = []
misclassified_samples = []
all_misclassified_samples = {}
best_accuracy = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {100 * correct/total:.2f}%")

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            misclassified_indices = (predicted != labels).nonzero(as_tuple=True)[0]
            for idx in misclassified_indices:
                file_path = val_dataset.samples[batch_idx * val_loader.batch_size + idx.item()][0]
                misclassified_samples.append({
                    "file_path": file_path,
                    "true_label": labels[idx].item(),
                    "predicted_label": predicted[idx].item()
                })
    all_misclassified_samples[f"epoch_{epoch+1}"] = misclassified_samples


    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    val_accuracy = 100 * val_correct / val_total
    accuracies.append(val_accuracy)

    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), 'best_mobilenet_v2.pth')

torch.save(model.state_dict(), 'final_mobilenet_v2.pth')

results = pd.DataFrame({
    "Epoch": list(range(1, num_epochs + 1)),
    "Train Loss": train_losses,
    "Validation Loss": val_losses,
    "Validation Accuracy": accuracies
})

results.to_csv("training_results.csv", index=False)

with open('misclassified_samples.json', 'w') as f:
    json.dump(misclassified_samples, f, indent=4)

hyperparameters = {
    "num_epochs": num_epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "optimizer": optimizer_type,
    "loss_function": loss_function,
    "train_dataset_size": len(train_dataset),
    "validation_dataset_size": len(val_dataset)
}

with open('training_info.json', 'w') as f:
    json.dump(hyperparameters, f, indent=4)



