import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

model_weights_path = 'best_model.pth'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dir = 'data_task/train'
test_dir = 'data_task/test/eagle/images - 2025-11-26T194151.195.jpg'

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = models.resnet50(pretrained=False)

model.load_state_dict(torch.load(model_weights_path, weights_only=False), strict=False)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)

model.train()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(num_epochs=10):
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train() 
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            
            optimizer.zero_grad()
            
            
            outputs = model(inputs)
            
            
            loss = criterion(outputs, labels)
            
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        
        print(f'Эпоха [{epoch+1}/{num_epochs}], Потеря: {epoch_loss:.4f}, Точность: {epoch_accuracy:.2f}%')
        
        accuracy = evaluate_model()
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Model saved with accuracy: {accuracy:.2f}%')
    
    print('Finished Training')

def evaluate_model():
    model.eval()  
    correct = 0
    total = 0
    
    with torch.no_grad(): 
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

#train_model(num_epochs=10)