import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import VGG
from dataloader import Cifar10Dataset
import torch.nn as nn

LEARNING_RATE = 0.05
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 30
NUM_WORKERS = 2


def train_fn(train_loader, model, optimizer, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Train Epoch : {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss : {loss.item():.6f}")


def train():
    model = VGG().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = Cifar10Dataset(train=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              shuffle=True)

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch [{epoch}/{EPOCHS}]")
        train_fn(train_loader, model, optimizer, epoch)

    torch.save(model.state_dict(), "./model.pt")


if __name__ == "__main__":
    train()