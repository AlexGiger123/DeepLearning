import torch
from torch import nn, optim
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

# Підготовка даних
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=128, shuffle=False)

# Визначення архітектури моделі LeNet-5
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(16, 120, kernel_size=5),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# Ініціалізація моделі, функції втрат та оптимізатора
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = LeNet5().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Тренування моделі
epochs = 10
for epoch in range(1, epochs + 1):
    total_loss = 0
    for batch, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        predictions = net(data)
        loss = loss_fn(predictions, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")

# Оцінка точності моделі
correct, total = 0, 0
net.eval()
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        predictions = net(data)
        _, predicted_labels = torch.max(predictions, 1)
        total += target.size(0)
        correct += (predicted_labels == target).sum().item()

print(f'Test Accuracy: {correct / total:.4f}')

# Функція для відображення зображень
def display_images(images, true_labels, predicted_labels, count=5):
    plt.figure(figsize=(10, 4))
    for idx in range(count):
        plt.subplot(1, count, idx + 1)
        image = images[idx].cpu().numpy().squeeze()
        plt.imshow(image, cmap='gray')
        plt.title(f"True: {true_labels[idx].item()}\nPred: {predicted_labels[idx].item()}")
        plt.axis('off')
    plt.show()

# Підготовка прикладів для візуалізації
sample_data = next(iter(test_loader))
sample_images, sample_labels = sample_data[0].to(device), sample_data[1]
sample_outputs = net(sample_images)
_, sample_predictions = torch.max(sample_outputs, 1)

# Відображення прикладів
display_images(sample_data[0], sample_labels, sample_predictions)
