import torch 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_data = datasets.MNIST(root="./data", train=True, download=False, transform=transform)
test_data = datasets.MNIST(root="./data", train=False, download=False, transform=transform)

train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return(x)
    
model = SimpleNet()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        images = images.view(images.shape[0], -1)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}")

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        images = images.view(images.shape[0], -1)
        output = model(images)
        prediction = torch.argmax(output, dim=1)
        correct += (prediction == labels).sum().item()
        total += labels.size(0)
        if labels.size(0):
            total = total + 1
    print(f"Accuracy: {correct / total:.1%}")