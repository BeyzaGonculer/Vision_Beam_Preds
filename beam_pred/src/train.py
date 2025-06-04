import torch
import torch.nn as nn

def train_model(model, dataloader, optimizer, device, num_epochs=5):
   
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

           
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

          
            loss.backward()
            optimizer.step()

           
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = total_correct / total_samples

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    print("Eğitim tamamlandı.")
