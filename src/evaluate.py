import os
import json
import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np

from data_loader import get_mnist_dataloaders
from model import CNN
from utils import set_seed

def evaluate_model(config_path="configs/config.yaml", checkpoint_path="checkpoints/best_model.pth"):
    # 1. Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Set seed
    set_seed(0)

    # 3. Device
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")

    # 4. Load test data
    _, _, test_loader = get_mnist_dataloaders(config)

    # 5. Load model
    model = CNN(num_classes=10).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 6. Collect predictions
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 7. Metrics
    print("Classification Report:\n")
    print(classification_report(all_labels, all_preds, digits=4))

        # 8. Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - MNIST")

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/confusion_matrix.png")
    print("✅ Confusion matrix saved at results/confusion_matrix.png")

    # 9. Save metrics as JSON
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "classification_report": classification_report(all_labels, all_preds, digits=4, output_dict=True)
    }

    os.makedirs("results", exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("✅ Metrics saved at results/metrics.json")


if __name__ == "__main__":
    evaluate_model()
