import torch
import torch.optim as optim
from ultralytics import YOLO
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import os

def modify_model_for_cifar10(model, debug=False):
    """Modified version to handle YOLOv8 cls model structure"""
    if hasattr(model, 'model'):
        backbone = model.model[:-1]

        if debug:
            print(f"Last backbone layer type: {type(backbone[-1]).__name__}")
        
            # Debug layer structure
            def print_layer_info(layer, prefix=''):
                for name, module in layer.named_children():
                    print(f"{prefix}Layer: {name}, Type: {type(module).__name__}")
                    if hasattr(module, 'conv') and hasattr(module.conv, 'out_channels'):
                        print(f"{prefix}  out_channels: {module.conv.out_channels}")
                    print_layer_info(module, prefix + '  ')
        
            print("Analyzing backbone structure:")
            print_layer_info(backbone[-1])
        
        # Get number of features from cv2 of C2f layer
        num_features = backbone[-1].cv2.conv.out_channels
        print(f"Number of features detected: {num_features}")
        
        # Create model with correct feature dimensions
        return nn.Sequential(
            backbone,  # Keep backbone as is
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(num_features, 10)
        )
    
    return model

def train_model(model, train_loader, val_loader, model_type, num_epochs=30, device='cuda'):
    """Train modified YOLO model on CIFAR-10"""
    # Create save directory if it doesn't exist
    save_dir = 'trained_models'
    os.makedirs(save_dir, exist_ok=True)
    
    best_model_path = os.path.join(save_dir, f'best_{model_type}_cifar10.pth')
    final_model_path = os.path.join(save_dir, f'final_{model_type}_cifar10.pth')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        train_loss = running_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        # Print progress
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Acc: {val_acc:.2f}%')
        
        # Save best model with model type
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
            }, best_model_path)
        
        scheduler.step(val_acc)
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'final_val_acc': val_acc,
    }, final_model_path)
    
    return model

def prepare_data():
    """Prepare CIFAR-10 dataloaders"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True,
                                    download=True, transform=transform)
    val_dataset = datasets.CIFAR10(root='./data', train=False,
                                  download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32,
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32,
                           shuffle=False, num_workers=2)
    
    return train_loader, val_loader

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    large_model = YOLO('yolov8l-cls.pt').model
    nano_model = YOLO('yolov8n-cls.pt').model
    
    # Modify for CIFAR-10
    print("Modifying models for CIFAR-10...")
    large_model = modify_model_for_cifar10(large_model)
    nano_model = modify_model_for_cifar10(nano_model)
    
    # Prepare data
    print("Preparing data...")
    train_loader, val_loader = prepare_data()
    
    # Train models with type identifiers
    print("\nTraining large model...")
    large_model = large_model.to(device)
    train_model(large_model, train_loader, val_loader, model_type='yolov8l', num_epochs=5, device=device)
    
    print("\nTraining nano model...")
    nano_model = nano_model.to(device)
    train_model(nano_model, train_loader, val_loader, model_type='yolov8n', num_epochs=2, device=device)

if __name__ == '__main__':
    main()