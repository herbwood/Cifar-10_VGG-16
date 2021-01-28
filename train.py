import torch
import torchvision.transforms as transforms
import torch.optim as optim
from model import VGG
from dataloader import Cifar10Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader

LEARNING_RATE = 0.05
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
WEIGHT_DECAY = 0
EPOCHS = 30
NUM_WORKERS = 2

def train(train_loader, model, optimizer):

    mean_loss = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        out = model(data)
        loss = F.cross_entropy(out, target)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Mean loss : {sum(mean_loss) / len(mean_loss)}")

def main():
    model = VGG().to(DEVICE)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = Cifar10Dataset(transform=transform)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              shuffle=True)

    for epoch in range(EPOCHS):
        print(f"Epoch [{epoch+1}/{EPOCHS}]")
        train(train_loader, model, optimizer)

    torch.save(model.state_dict(), "./model.pt")

if __name__ == "__main__":
    main()