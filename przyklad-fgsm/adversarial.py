import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. Przygotowanie danych
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

# 2. Prosty model sieci neuronowej
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 3. Trenowanie modelu
def train_model(model, train_loader, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

# 4. Testowanie dokładności
def test_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            output = model(images)
            _, predicted = torch.max(output, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    print(f"Accuracy: {correct / total:.2%}")

# 5. Atak FGSM
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, 0, 1)

def test_adversarial(model, device, test_loader, epsilon):
    correct = 0
    adv_examples = []

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() != target.item():
            continue

        loss = F.cross_entropy(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data

        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]

        if final_pred.item() == target.item():
            correct += 1
        if len(adv_examples) < 5:
            adv_examples.append((init_pred.item(), final_pred.item(), perturbed_data.squeeze().detach().cpu().numpy()))

    final_acc = correct / len(test_loader)
    print(f"FGSM Attack (ε={epsilon}) Accuracy: {final_acc:.2%}")
    return adv_examples

# 6. Wykonanie wszystkiego
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNN().to(device)
train_model(model, train_loader)
test_model(model, test_loader)

# 7. Atak
epsilon = 0.25
examples = test_adversarial(model, device, test_loader, epsilon)

# 8. Wizualizacja kilku przykładów
def show_examples(examples):
    plt.figure(figsize=(10,5))
    for i, (orig, adv, img) in enumerate(examples):
        plt.subplot(1, len(examples), i+1)
        plt.xticks([])
        plt.yticks([])
        plt.title(f"{orig} → {adv}")
        plt.imshow(img, cmap="gray")
    plt.show()

show_examples(examples)
