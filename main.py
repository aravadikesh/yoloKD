import torch
import torch.nn as nn
import torch.optim as optim
from ultralytics import YOLO
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from modify_and_train import modify_model_for_cifar10
import copy

# Load teacher and student models from CIFAR-10 trained checkpoints
teacher_model = YOLO("yolov8l-cls.pt").model  # Base architecture
student_model = YOLO("yolov8n-cls.pt").model  # Base architecture

# Modify models for CIFAR-10
teacher_model = modify_model_for_cifar10(teacher_model)
student_model = modify_model_for_cifar10(student_model)

# Load trained CIFAR-10 weights
teacher_checkpoint = torch.load('trained_models/best_yolov8l_cifar10.pth')
student_checkpoint = torch.load('trained_models/best_yolov8n_cifar10.pth')

teacher_model.load_state_dict(teacher_checkpoint['model_state_dict'])
student_model.load_state_dict(student_checkpoint['model_state_dict'])

# Set model modes
teacher_model = teacher_model.eval()  # Teacher in eval mode
student_model = student_model.train()  # Student in training mode

# Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model = teacher_model.to(device)
student_model = student_model.to(device)

# Utility function for logit standardization
def standardize_logits(logits):
    mean = logits.mean(dim=-1, keepdim=True)
    std = logits.std(dim=-1, keepdim=True)
    standardized_logits = (logits - mean) / (std + 1e-8)
    return standardized_logits

# Function to adjust temperature over epochs (Curriculum Distillation)
def adjust_temperature(epoch, initial_temp=5.0, min_temp=1.0):
    return max(min_temp, initial_temp * (0.8 ** epoch))

# 2. Define Knowledge Distillation Loss with Logit Standardization
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.3, temperature=3.0):  # Reduce alpha from 0.5
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

def model_forward(model, x):
    """Wrapper to handle YOLO model outputs"""
    output = model(x)
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output

# In main.py
def train_student_with_distillation(teacher_model, student_model, dataloader, num_epochs=10, alpha=0.5, initial_temperature=5.0):
    # Load pretrained teacher model
    teacher_checkpoint = torch.load('trained_models/best_yolov8l_cifar10.pth')
    teacher_model.load_state_dict(teacher_checkpoint['model_state_dict'])
    teacher_model.eval()
    
    criterion = DistillationLoss(alpha=alpha, temperature=initial_temperature)
    optimizer = optim.Adam(student_model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
    
    for epoch in range(num_epochs):
        temperature = adjust_temperature(epoch, initial_temperature)
        criterion.temperature = temperature
        running_loss = 0.0

        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            with torch.no_grad():
                teacher_logits = model_forward(teacher_model, images)
            
            student_logits = model_forward(student_model, images)
            student_loss = nn.CrossEntropyLoss()(student_logits, labels)
            
            # Knowledge distillation loss
            loss = criterion(student_logits, teacher_logits.detach(), labels, student_loss)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i}/{len(dataloader)}], "
                      f"Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(dataloader)
        scheduler.step(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
        
    # Save final distilled model
    torch.save(student_model.state_dict(), "distilled_yolov8_student.pth")
    print("Distilled Student Model Saved.")
    return student_model

# 4. Prepare CIFAR-10 Dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # More reasonable size for classification
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Download and load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Execute Training
train_student_with_distillation(
    teacher_model=teacher_model,
    student_model=student_model, 
    dataloader=train_dataloader,
    num_epochs=10,
    alpha=0.5,
    initial_temperature=5.0
)