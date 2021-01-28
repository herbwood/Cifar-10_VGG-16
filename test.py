import torch
import torchvision.transforms as transforms
import torch.optim as optim
from model import VGG
from dataloader import Cifar10Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader

LEARNING_RATE = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256
WEIGHT_DECAY = 0
NUM_WORKERS = 2

def evalute(model, test_loader):
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)

            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum.item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    return test_loss, test_accuracy

def main():
    model = VGG()
    model.load_state_dict(torch.load("./model.pt"))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = Cifar10Dataset(train=False, transform=transform)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE,
                             num_workers=NUM_WORKERS,
                             shuffle=False)

    test_loss, test_accuracy = evalute(model, test_loader)
    print(test_loss, test_accuracy)

if __name__ == "__main__":
    main()


