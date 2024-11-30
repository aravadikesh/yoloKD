import torch
import torch.nn as nn
import torch.optim as optim
from ultralytics import YOLO
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from modify_and_train import get_dataset, modify_model_for_cifar10, modify_model_for_dataset
import copy
from sklearn.model_selection import ParameterGrid
import json
from datetime import datetime
import argparse

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
# class SelfDistillationWrapper:
#     def __init__(self, student_model, alpha=0.3):
#         self.student = student_model
#         self.pseudo_teacher = copy.deepcopy(student_model)
#         self.alpha = alpha
    
#     def update_pseudo_teacher(self):
#         self.pseudo_teacher.load_state_dict(self.student.state_dict())

#     def self_distillation_loss(self, student_logits, pseudo_teacher_logits):
#         return torch.nn.functional.kl_div(
#             torch.nn.functional.log_softmax(student_logits, dim=-1),
#             torch.nn.functional.softmax(pseudo_teacher_logits, dim=-1),
#             reduction="batchmean"
#         )

def model_forward(model, x):
    """Wrapper to handle YOLO model outputs"""
    output = model(x)
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output

def validate_model(model, val_loader, criterion, device):
    """Validate model performance"""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    accuracy = 100. * correct / total
    return accuracy

def grid_search_distillation(teacher_model, base_student_model, train_subset, val_subset, device, num_epochs=3):
    """Perform grid search for best hyperparameters"""
    param_grid = {
        'alpha': [0.3, 0.5, 0.7],
        'initial_temperature': [3.0, 5.0, 7.0],
        'learning_rate': [0.0001, 0.001, 0.01],
        'batch_size': [16, 32, 64]
    }
    
    results = []
    best_accuracy = 0
    best_params = None
    
    for params in ParameterGrid(param_grid):
        print(f"\nTrying parameters: {params}")

        # Prepare DataLoaders
        train_loader = DataLoader(train_subset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=params['batch_size'], shuffle=False)
        
        # Reset student model
        student_model = copy.deepcopy(base_student_model)
        student_model.train()
        
        # Training with current parameters
        criterion = DistillationLoss(alpha=params['alpha'], 
                                   temperature=params['initial_temperature'])
        optimizer = optim.Adam(student_model.parameters(), lr=params['learning_rate'])
        
        # Quick training (reduced epochs for grid search)
        for epoch in range(num_epochs):
            running_loss = 0.0
            for  i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                
                with torch.no_grad():
                    teacher_logits = model_forward(teacher_model, images)
                
                student_logits = model_forward(student_model, images)
                student_loss = nn.CrossEntropyLoss()(student_logits, labels)
                
                loss = criterion(student_logits, teacher_logits.detach(), 
                               labels, student_loss)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if i % 100 == 0:
                    print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i}/{len(train_loader)}], "
                        f"Loss: {loss.item():.4f}")
        
        # Validate
        accuracy = validate_model(student_model, val_loader, criterion, device)
        
        results.append({
            'params': params,
            'accuracy': accuracy
        })
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
            
        print(f"Validation Accuracy: {accuracy:.2f}%")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'grid_search_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\nBest parameters found:")
    print(json.dumps(best_params, indent=4))
    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    
    return best_params

# In main.py
def train_student_with_distillation(teacher_model, student_model, dataset, num_epochs=10, alpha=0.5, initial_temperature=5.0, learning_rate=0.001, batch_size=32, device="cuda"):
    # Prepare DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    teacher_model.eval()
    
    criterion = DistillationLoss(alpha=alpha, temperature=initial_temperature)
    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)
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
        
    # # Save final distilled model
    # torch.save(student_model.state_dict(), "distilled_yolov8_student.pth")
    # print("Distilled Student Model Saved.")
    return student_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'pets'])
    args = parser.parse_args()

    # Load dataset
    train_dataset, test_dataset, num_classes = get_dataset(args.dataset)
    
    # Load and modify models for dataset
    teacher_model = modify_model_for_dataset(YOLO("yolov8l-cls.pt").model, num_classes)
    student_model = modify_model_for_dataset(YOLO("yolov8n-cls.pt").model, num_classes)

    # Load trained weights
    teacher_checkpoint = torch.load(f'trained_models/best_yolov8l_{args.dataset}.pth')
    student_checkpoint = torch.load(f'trained_models/best_yolov8n_{args.dataset}.pth')
    
    teacher_model.load_state_dict(teacher_checkpoint['model_state_dict'])
    student_model.load_state_dict(student_checkpoint['model_state_dict'])

    # Set model modes
    teacher_model = teacher_model.eval()  # Teacher in eval mode
    student_model = student_model.train()  # Student in training mode

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)

    # 4. Prepare CIFAR-10 Dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # More reasonable size for classification
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Prepare validation set
    dataset_size = len(train_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    # # Perform grid search
    # best_params = grid_search_distillation(
    #     teacher_model, 
    #     student_model,
    #     train_subset,
    #     val_subset,
    #     device
    # )

    # # Update training parameters with best found
    # alpha = best_params['alpha']
    # initial_temperature = best_params['initial_temperature']
    # learning_rate = best_params['learning_rate']
    # batch_size = best_params['batch_size']

    # # Execute training with best parameters
    # train_student_with_distillation(
    #     teacher_model=teacher_model,
    #     student_model=student_model, 
    #     dataset=train_dataset,
    #     num_epochs=10,
    #     alpha=alpha,
    #     initial_temperature=initial_temperature,
    #     learning_rate=learning_rate,
    #     batch_size=batch_size,
    # )

    # Execute Training
    student_model = train_student_with_distillation(
        teacher_model=teacher_model,
        student_model=student_model, 
        dataset=train_dataset,
        num_epochs=10,
        alpha=1,
        initial_temperature=3.0,
        learning_rate=0.001,
        batch_size=32,
    )

    # Save final distilled model
    torch.save(student_model.state_dict(), f'distilled_yolov8n_{args.dataset}.pth')

if __name__ == "__main__":
    main()