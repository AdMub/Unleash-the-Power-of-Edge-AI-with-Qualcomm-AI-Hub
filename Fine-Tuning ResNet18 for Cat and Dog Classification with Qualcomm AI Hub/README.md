# Fine-Tuning ResNet18 for Cat and Dog Classification with Qualcomm AI Hub

(![Project Banner](https://github.com/user-attachments/assets/5be19659-8869-42ed-81cb-20085b632238)
)

## 📌 Project Overview
This project fine-tunes the ResNet18 model to classify images of cats and dogs using the Microsoft Kaggle Cats and Dogs dataset. The model is trained using PyTorch and optimized for Qualcomm AI Hub deployment.

![AI-themed illustration](https://github.com/user-attachments/assets/b4f8ccbc-ca94-45c8-b96d-1b2fc20a8135)


## 🚀 Installation

### 1️⃣ Install the necessary libraries
```bash
pip install qai-hub
qai-hub configure --api_token YOUR_API_TOKEN
pip install qai-hub qai-hub-models
```

### 2️⃣ Download and Clean the Dataset
```bash
curl -O https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip
unzip kagglecatsanddogs_5340.zip -d PetImages
ls PetImages  # Should list 'Cat' and 'Dog' directories
```

#### Remove Corrupted Images
```python
import os

num_skipped = 0
for folder_name in ("Cat", "Dog"):
    folder_path = os.path.join("PetImages", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = b"JFIF" in fobj.peek(10)
        finally:
            fobj.close()
        if not is_jfif:
            num_skipped += 1
            os.remove(fpath)

print(f"Deleted {num_skipped} images.")
```

## 📦 Data Preprocessing
We apply data transformations to improve generalization.
```python
from torchvision import transforms, datasets

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder("PetImages", transform=data_transforms)
```

## 🔥 Model Training
We fine-tune the ResNet18 model, training only the final layer.
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Load Pretrained Model
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 2)  # Two classes: Cat and Dog

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
```

### Training Loop
```python
def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs} completed.")
    return model
```

## 🎯 Model Testing
Predict a random image from the dataset.
```python
import random
import matplotlib.pyplot as plt

def predict_random_image(model, dataset):
    idx = random.randint(0, len(dataset) - 1)
    img, label = dataset[idx]
    with torch.no_grad():
        output = model(img.unsqueeze(0))
        _, predicted = torch.max(output, 1)
    class_names = dataset.classes
    plt.imshow(transforms.ToPILImage()(img))
    plt.title(f'Predicted: {class_names[predicted]}')
    plt.axis('off')
    plt.show()
```

## 📲 Qualcomm AI Hub Deployment
```python
import qai_hub as hub

selected_device = "Samsung Galaxy S22 Ultra 5G"
device_model = models.resnet18(pretrained=False)
device_model.fc = nn.Linear(device_model.fc.in_features, 2)

torch.save(model.state_dict(), "cats_and_dogs_model.pth")
device_model.load_state_dict(torch.load("cats_and_dogs_model.pth"))
device_model.eval()

scripted_model = torch.jit.script(device_model)
scripted_model.save("cats_and_dogs_scripted.pt")

uploaded_model = hub.upload_model(scripted_model, name="Cats_and_Dogs_Model")
compile_job = hub.submit_compile_job(uploaded_model, device=hub.Device(selected_device))
```

## 📜 Results
| Epoch | Train Loss | Train Accuracy | Valid Loss | Valid Accuracy |
|-------|-----------|---------------|------------|---------------|
| 1     | 0.0854    | 96.78%        | 0.0328     | 98.74%        |
| 2     | 0.0510    | 98.16%        | 0.0322     | 98.89%        |
| 3     | 0.0462    | 98.32%        | 0.0290     | 98.85%        |
| 4     | 0.0303    | 98.84%        | 0.0226     | 99.34%        |
| 5     | 0.0297    | 99.02%        | 0.0202     | 99.40%        |

## 📌 Conclusion
The fine-tuned ResNet18 model achieved **99.4% validation accuracy** on the cats and dogs dataset. The model was further compiled and optimized for Qualcomm AI Hub, making it ready for mobile deployment.

## 🛠 Future Improvements
- Extend dataset with more animal categories
- Experiment with deeper architectures like ResNet50 or EfficientNet
- Implement real-time inference on Qualcomm AI devices

## 🔗 References
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Torchvision Models](https://pytorch.org/vision/stable/models.html)
- [Qualcomm AI Hub](https://www.qualcomm.com/research/artificial-intelligence)

## ✨ Acknowledgments
Special thanks to Qualcomm AI Hub for providing model deployment support and Microsoft for the dataset.

---
_@ StackUp_



