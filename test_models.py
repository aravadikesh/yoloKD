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
import argparse
from modify_and_train import modify_model_for_dataset, get_dataset, get_num_classes

def load_trained_model(model_type, dataset_name, device):
    """Load trained model from saved weights"""
    base_model = YOLO(f"yolov8{model_type}-cls.pt").model
    _, _, num_classes = get_dataset(dataset_name)
    model = modify_model_for_dataset(base_model, num_classes).to(device)
    
    checkpoint = torch.load(f'trained_models/best_yolov8{model_type}_{dataset_name}.pth')
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
    n_classes = len(class_names)
    scale = 0.8*n_classes  # Scale for better readability
    figsize = (3*scale, scale)
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    for idx, (name, result) in enumerate(results.items()):
        cm = confusion_matrix(result['targets'], result['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx],
                   xticklabels=class_names, yticklabels=class_names)
        axes[idx].set_title(f'{name} Confusion Matrix')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')
        if n_classes > 10:  # Rotate labels for better readability
            plt.setp(axes[idx].get_xticklabels(), rotation=45, ha='right')
            plt.setp(axes[idx].get_yticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def load_distilled_model(dataset_name, device):
    """Load distilled model from saved weights"""
    base_model = YOLO("yolov8n-cls.pt").model
    model = modify_model_for_dataset(base_model, get_num_classes(dataset_name)).to(device)
    
    checkpoint = torch.load(f'distilled_yolov8n_{dataset_name}.pth')
    model.load_state_dict(checkpoint)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'pets'])
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    _, test_dataset, _ = get_dataset(args.dataset)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load models
    models = {}
    for model_type in ['l', 'n']:  # large and nano
        model, val_acc = load_trained_model(model_type, args.dataset, device)
        models[f'YOLOv8{"Large" if model_type == "l" else "Nano"}'] = model
        print(f"Loaded YOLOv8{model_type} (Validation Accuracy: {val_acc:.2f}%)")
    
    # Load distilled model
    distilled_model = load_distilled_model(args.dataset, device)
    models['Distilled-Nano'] = distilled_model
    print("Loaded Distilled Nano")
    
    # Evaluate models
    results = {}
    class_names = (test_dataset.classes if hasattr(test_dataset, 'classes') 
                  else [str(i) for i in range(len(set(test_dataset.targets)))])
    
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
            target_names=class_names
        ))
    
    # Plot confusion matrices
    save_path = f'confusion_matrices_{args.dataset}.png'
    plot_confusion_matrix(results, class_names, save_path)
    print(f"\nResults saved to {save_path}")

if __name__ == "__main__":
    main()