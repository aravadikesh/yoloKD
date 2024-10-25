import torch
import torch.nn as nn
import torch.optim as optim
from ultralytics import YOLO

# 1. Load Teacher and Student Models
# Load a pre-trained larger YOLOv8 model as the teacher
teacher_model = YOLO("yolov8l.pt").model.eval()  # Using YOLOv8 Large (L) as Teacher

# Load a smaller YOLOv8 model as the student
student_model = YOLO("yolov8n.pt").model.train()  # Using YOLOv8 Nano (N) as Student

# 2. Define Knowledge Distillation Loss
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=3.0):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha  # Balance weight between student and distillation loss
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, true_labels, student_loss):
        # Soft target (distillation) loss using KL Divergence
        teacher_probs = nn.functional.softmax(teacher_logits / self.temperature, dim=1)
        student_probs = nn.functional.log_softmax(student_logits / self.temperature, dim=1)
        distillation_loss = self.kl_div(student_probs, teacher_probs) * (self.temperature ** 2)
        
        # Total Loss: combination of distillation and student losses
        return (1 - self.alpha) * student_loss + self.alpha * distillation_loss

# 3. Set Up Training Loop with Knowledge Distillation
def train_student_with_distillation(teacher_model, student_model, dataloader, num_epochs=10, alpha=0.5, temperature=3.0):
    # Initialize distillation loss function
    criterion = DistillationLoss(alpha=alpha, temperature=temperature)
    optimizer = optim.Adam(student_model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            # Teacher Model Forward Pass
            with torch.no_grad():  # Freeze teacher weights
                teacher_logits = teacher_model(images)

            # Student Model Forward Pass
            student_logits = student_model(images)

            # Calculate Student Loss
            student_loss = nn.CrossEntropyLoss()(student_logits, labels)

            # Calculate Distillation Loss
            loss = criterion(student_logits, teacher_logits, labels, student_loss)
            
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

# 4. Prepare the Dataset and DataLoader
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCDetection
import yaml
import os
from PIL import Image

# Load the VOC.yaml file to read dataset paths and configuration
yaml_path = "/mnt/data/VOC.yaml"
with open(yaml_path, 'r') as f:
    voc_config = yaml.safe_load(f)

# Paths to VOC data
voc_root = voc_config['path']
train_list = os.path.join(voc_root, voc_config['train'])
val_list = os.path.join(voc_root, voc_config['val'])
num_classes = voc_config['nc']
class_names = voc_config['names']

# Define transformations for YOLO input requirements
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Resize to YOLOv8's input size
    transforms.ToTensor()
])

# Custom Dataset Class for VOC to handle paths from train.txt
class CustomVOCDataset(VOCDetection):
    def __init__(self, root, image_set, transform=None, target_transform=None):
        super(CustomVOCDataset, self).__init__(root, year="2012", image_set=image_set,
                                               transform=transform, target_transform=target_transform)
        # Read image paths
        with open(image_set, 'r') as f:
            self.image_paths = [os.path.join(root, 'JPEGImages', line.strip() + '.jpg') for line in f]
        self.transform = transform

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # For simplicity, return a dummy target (replace with actual processing if needed)
        target = []  # Load actual target if needed
        return image, target

    def __len__(self):
        return len(self.image_paths)

# Initialize Dataset and DataLoader
train_dataset = CustomVOCDataset(root=voc_root, image_set=train_list, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model, student_model = teacher_model.to(device), student_model.to(device)

# Training Execution
train_student_with_distillation(teacher_model, student_model, train_dataloader, num_epochs=10, alpha=0.5, temperature=3.0)

# Save the Distilled Student Model
torch.save(student_model.state_dict(), "distilled_yolov8_student.pth")
print("Distilled Student Model Saved.")