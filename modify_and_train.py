import torch
import torch.optim as optim
from ultralytics import YOLO
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import os
import argparse

def modify_model_for_dataset(model, num_classes, debug=False):
    """Modify model for specific dataset"""
    if hasattr(model, 'model'):
        backbone = model.model[:-1]
        num_features = backbone[-1].cv2.conv.out_channels
        
        return nn.Sequential(
            backbone,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes)
        )
    return model

def train_model(model, train_loader, val_loader, model_type, dataset_name='cifar10', num_epochs=30, device='cuda'):
    """Train modified YOLO model on CIFAR-10"""
    # Create save directory if it doesn't exist
    save_dir = 'trained_models'
    os.makedirs(save_dir, exist_ok=True)
    
    best_model_path = os.path.join(save_dir, f'best_{model_type}_{dataset_name}.pth')
    final_model_path = os.path.join(save_dir, f'final_{model_type}_{dataset_name}.pth')
    
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
        
        for i, (images, labels) in enumerate(train_loader):
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
            
            if i % 1000 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i}/{len(train_loader)}], "
                        f"Loss: {loss.item():.4f}")
        
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

def get_num_classes(dataset_name='cifar10'):
    if dataset_name.lower() == 'cifar10':
        return 10
    elif dataset_name.lower() == 'pets':
        return 37
    elif dataset_name.lower() == 'tiny-imagenet':
        return 200
    return 10

def get_dataset(dataset_name='cifar10'):
    """Prepare dataset transformations and loading"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    if dataset_name.lower() == 'cifar10':
        train_dataset = datasets.CIFAR10(root='./data', train=True,
                                       download=True, transform=transform)
        val_dataset = datasets.CIFAR10(root='./data', train=False,
                                     download=True, transform=transform)
        num_classes = 10
        
    elif dataset_name.lower() == 'pets':
        train_dataset = datasets.OxfordIIITPet(
            root='./data', split='trainval',
            download=True, transform=transform
        )
        val_dataset = datasets.OxfordIIITPet(
            root='./data', split='test',
            download=True, transform=transform
        )
        num_classes = 37  # 37 pet breeds
        
    elif dataset_name.lower() == 'tiny-imagenet':        
        # Restructure validation data
        tiny_imagenet_path = './data/tiny-imagenet-200'
        train_dir = os.path.join(tiny_imagenet_path, 'train')
        val_dir = os.path.join(tiny_imagenet_path, 'val')
        
        # Create proper validation structure
        if not os.path.exists(os.path.join(val_dir, 'images')):
            print("Restructuring validation data...")
            val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')
            
            # Read annotations
            with open(val_annotations_file, 'r') as f:
                val_annotations = f.readlines()
            
            # Create class directories
            for line in val_annotations:
                parts = line.strip().split('\t')
                img_name, class_id = parts[0], parts[1]
                
                # Make class directory
                class_dir = os.path.join(val_dir, class_id)
                os.makedirs(class_dir, exist_ok=True)
                
                # Move image to class directory
                old_img_path = os.path.join(val_dir, 'images', img_name)
                new_img_path = os.path.join(class_dir, img_name)
                if os.path.exists(old_img_path):
                    os.rename(old_img_path, new_img_path)
        
        train_dataset = datasets.ImageFolder(train_dir, transform=transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=transform)
        num_classes = 200
        
    return train_dataset, val_dataset, num_classes

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'pets', 'tiny-imagenet'])
    args = parser.parse_args()
    
    # Load dataset
    train_dataset, val_dataset, num_classes = get_dataset(args.dataset)
    
    # Load and modify models
    large_model = YOLO('yolov8l-cls.pt').model
    nano_model = YOLO('yolov8n-cls.pt').model
    
    large_model = modify_model_for_dataset(large_model, num_classes)
    nano_model = modify_model_for_dataset(nano_model, num_classes)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Train models with type identifiers
    print(f"\nTraining on {args.dataset} dataset...")
    print("\nTraining large model...")
    large_model = large_model.to(device)
    train_model(large_model, train_loader, val_loader, model_type='yolov8l', dataset_name=args.dataset, num_epochs=5, device=device)
    
    print("\nTraining nano model...")
    nano_model = nano_model.to(device)
    train_model(nano_model, train_loader, val_loader, model_type='yolov8n', dataset_name=args.dataset, num_epochs=2, device=device)

if __name__ == '__main__':
    main()