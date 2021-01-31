import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from model import VGG
from dataloader import Cifar10Dataset
import argparse

parser = argparse.ArgumentParser(description="VGG Training")
parser.add_argument("--epochs", default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument("--lr", '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='inital learning rate')
parser.add_argument("--batch", "--batch-size", default=128, type=int,
                    metavar="B", help='mini-batch size')
parser.add_argument('--workers', '--num-workers', default=2, type=int, metavar='W',
                    help='number of data loading workers')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    args = parser.parse_args()

    model = VGG().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = Cifar10Dataset(train=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch,
                              num_workers=args.workers,
                              shuffle=True)

    leastloss = 1e+5

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch [{epoch}/{args.epochs}]")
        loss = train_fn(train_loader, model, optimizer, criterion, epoch)
        scheduler.step()

        if loss < leastloss:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
            }, 'model.pt')

            leastloss = loss

            print(f"saved model with loss : {leastloss:.4f}")

if __name__ == "__main__":
    # python train.py --epochs 50 --batch 128 --lr 0.0001 --workers 2
    train()