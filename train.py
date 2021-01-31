import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import VGG
from dataloader import Cifar10Dataset
import torch.nn as nn

LEARNING_RATE = 0.0001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 30
NUM_WORKERS = 2


def train_fn(train_loader, model, optimizer, criterion, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            output = model(data)
            loss = criterion(output, target)

            pred = output.max(1, keepdim=True)[1]
            correct = pred.eq(target.view_as(pred)).sum().item()

            loss.backward()
            optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Train Epoch : {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]"
                  f"\tLoss : {loss.item():.6f}"
                  f"\tAccuracy : {(100. * correct / train_loader.batch_size):.2f}%")

    return loss

def train():
    model = VGG().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = Cifar10Dataset(train=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              shuffle=True)

    leastloss = 1e+5

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch [{epoch}/{EPOCHS}]")
        loss = train_fn(train_loader, model, optimizer, criterion, epoch)
        scheduler.step()

        if loss < leastloss:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
            }, f'model_epoch_{epoch}_loss_{loss}.pt')

            leastloss = loss

            print(f"saved model with loss : {leastloss:.4f}")

if __name__ == "__main__":
    train()