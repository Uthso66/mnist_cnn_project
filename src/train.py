import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import yaml
import os

from data_loader import get_mnist_dataloaders
from model import CNN
from utils import set_seed

def train(config_path="configs/config.yaml"):
    # 1. Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Set seed for reproducibility
    set_seed(0)

    # 3. Device
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")

    # 4. Dataloaders
    train_loader, val_loader, test_loader = get_mnist_dataloaders(config)

    # 5. Model, loss, optimizer
    model = CNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # 6. Training loop
    best_val_acc = 0.0
    for epoch in range(config['training']['epochs']):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}", leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loop.set_postfix(loss=loss.item())

        train_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}] Training Loss: {running_loss/len(train_loader):.4f}, Training Acc: {train_acc:.2f}%")

        # 7. Validation step
        val_acc = evaluate(model, val_loader, device)
        print(f"Validation Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print("✅ Saved new best model!")

    print("Training complete.")

def evaluate(model, dataloader, device):
    """
    Runs evaluation on validation data.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

if __name__ == "__main__":
    train()
