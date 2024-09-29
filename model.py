import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset, concatenate_datasets
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import numpy as np

# Load the dataset from Hugging Face
dataset = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets")

# Select the first 10,000 and last 10,000 images from the 'train' split
first_10000 = dataset['train'].select(range(10000))
last_10000 = dataset['train'].select(range(len(dataset['train']) - 10000, len(dataset['train'])))

# Combine both subsets using concatenate_datasets
combined_dataset = concatenate_datasets([first_10000, last_10000])

# Preprocess the Data
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 for ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Dataset class to handle image loading and preprocessing
class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']  # Get image data
        label = self.dataset[idx]['label']  # Label (0 for real, 1 for AI-generated)

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # If the image is grayscale, convert it to RGB
        if image.mode != 'RGB':
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# Create train dataset and dataloader
train_data = ImageDataset(combined_dataset, transform=image_transforms)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Define the Classification Model (ResNet)
class ImageClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(ImageClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Initialize the model, loss function, and optimizer
model = ImageClassifier(num_classes=2)
criterion = nn.CrossEntropyLoss()  # Binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training Function
def train_model(model, criterion, optimizer, train_loader, epochs=1):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Training loop
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Track accuracy and loss
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        train_loss = running_loss / len(train_loader)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

# Train the Model
train_model(model, criterion, optimizer, train_loader, epochs=1)

# Save the Model
torch.save(model.state_dict(), 'resnet18_ai_vs_real.pth')
