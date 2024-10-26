import torch
import torch.nn as nn
import torch.optim as optim
from ultralytics import YOLO
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
import os
import copy

# 1. Load Teacher and Student Models
teacher_model = YOLO("yolov8l.pt").model.eval()  # Large as teacher
student_model = YOLO("yolov8n.pt").model.train()  # Nano as student

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model, student_model = teacher_model.to(device), student_model.to(device)

# Utility function for logit standardization
def standardize_logits(logits):
    mean = logits.mean(dim=-1, keepdim=True)
    std = logits.std(dim=-1, keepdim=True)
    standardized_logits = (logits - mean) / (std + 1e-8)
    return standardized_logits

# Function to adjust temperature over epochs (Curriculum Distillation)
def adjust_temperature(epoch, initial_temp=5.0, decay=0.95):
    return max(1.0, initial_temp * (decay ** epoch))

# 2. Define Knowledge Distillation Loss with Logit Standardization
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=3.0):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, true_labels, student_loss):
        teacher_logits = standardize_logits(teacher_logits)  # Apply standardization
        teacher_probs = nn.functional.softmax(teacher_logits / self.temperature, dim=1)
        student_probs = nn.functional.log_softmax(student_logits / self.temperature, dim=1)
        distillation_loss = self.kl_div(student_probs, teacher_probs) * (self.temperature ** 2)
        return (1 - self.alpha) * student_loss + self.alpha * distillation_loss

# Self-distillation Wrapper for student model
class SelfDistillationWrapper:
    def __init__(self, student_model, alpha=0.3):
        self.student = student_model
        self.pseudo_teacher = copy.deepcopy(student_model)
        self.alpha = alpha
    
    def update_pseudo_teacher(self):
        self.pseudo_teacher.load_state_dict(self.student.state_dict())

    def self_distillation_loss(self, student_logits, pseudo_teacher_logits):
        return torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(student_logits, dim=-1),
            torch.nn.functional.softmax(pseudo_teacher_logits, dim=-1),
            reduction="batchmean"
        )

# 3. Training Loop with Knowledge Distillation, Curriculum Temperature, Self-Distillation
def train_student_with_distillation(teacher_model, student_model, dataloader, num_epochs=10, alpha=0.5, initial_temperature=5.0):
    criterion = DistillationLoss(alpha=alpha, temperature=initial_temperature)
    optimizer = optim.Adam(student_model.parameters(), lr=0.001)
    self_distiller = SelfDistillationWrapper(student_model, alpha=0.3)

    for epoch in range(num_epochs):
        temperature = adjust_temperature(epoch, initial_temperature)
        criterion.temperature = temperature
        running_loss = 0.0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            with torch.no_grad():
                teacher_logits = teacher_model(images)
            
            student_logits = student_model(images)
            student_loss = nn.CrossEntropyLoss()(student_logits, labels)
            kd_loss = criterion(student_logits, teacher_logits, labels, student_loss)
            
            pseudo_teacher_logits = self_distiller.pseudo_teacher(images)
            sd_loss = self_distiller.self_distillation_loss(student_logits, pseudo_teacher_logits)
            
            loss = (1 - self_distiller.alpha) * kd_loss + self_distiller.alpha * sd_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        self_distiller.update_pseudo_teacher()  # Update pseudo-teacher
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

# 4. Prepare CIFAR-10 Dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Resize CIFAR-10 to YOLO's input size
    transforms.ToTensor()
])

# Download and load CIFAR-10 dataset
train_dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Execute Training
train_student_with_distillation(teacher_model, student_model, train_dataloader, num_epochs=10, alpha=0.5)

# Save Model
torch.save(student_model.state_dict(), "distilled_yolov8_student.pth")
print("Distilled Student Model Saved.")
