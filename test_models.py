# test_models.py
import torch
import time
from ultralytics import YOLO
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from modify_and_train import modify_model_for_cifar10

def load_trained_model(model_type, device):
    """Load trained model from saved weights"""
    base_model = YOLO(f"yolov8{model_type}-cls.pt").model
    model = modify_model_for_cifar10(base_model).to(device)
    
    checkpoint = torch.load(f'trained_models/best_yolov8{model_type}_cifar10.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint.get('val_acc', 0)

def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = []
    targets = []
    inference_times = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            start_time = time.time()
            outputs = model(images)
            inference_time = time.time() - start_time
            
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.numpy())
            inference_times.append(inference_time)
    
    return {
        'predictions': predictions,
        'targets': targets,
        'avg_inference_time': np.mean(inference_times)
    }

def plot_confusion_matrix(results, class_names, save_path='confusion_matrices.png'):
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))  # Changed to 3 columns
    
    for idx, (name, result) in enumerate(results.items()):
        cm = confusion_matrix(result['targets'], result['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx],
                   xticklabels=class_names, yticklabels=class_names)
        axes[idx].set_title(f'{name} Confusion Matrix')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def load_distilled_model(device):
    """Load distilled model from saved weights"""
    base_model = YOLO("yolov8n-cls.pt").model
    model = modify_model_for_cifar10(base_model).to(device)
    
    checkpoint = torch.load('distilled_yolov8_student.pth')
    model.load_state_dict(checkpoint)
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare test data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load trained models including distilled
    models = {}
    for model_type in ['l', 'n']:  # large and nano
        model, val_acc = load_trained_model(model_type, device)
        models[f'YOLOv8{"Large" if model_type == "l" else "Nano"}'] = model
        print(f"Loaded YOLOv8{model_type} (Validation Accuracy: {val_acc:.2f}%)")
    
    # Load distilled model
    print("\nLoading distilled model...")
    distilled_model = load_distilled_model(device)
    models['Distilled-Nano'] = distilled_model
    
    # Evaluate all models
    results = {}
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        result = evaluate_model(model, test_loader, device)
        results[name] = result
        
        accuracy = np.mean(np.array(result['targets']) == np.array(result['predictions']))
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Average Inference Time: {result['avg_inference_time']:.4f} seconds")
        print("\nClassification Report:")
        print(classification_report(
            result['targets'],
            result['predictions'],
            target_names=test_dataset.classes
        ))
    
    # Plot confusion matrices for all three models
    plot_confusion_matrix(results, test_dataset.classes)
    print("\nResults saved to confusion_matrices.png")

if __name__ == "__main__":
    main()