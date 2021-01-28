import torch
import torch.nn as nn
import torch.nn.functional as F

architecture_config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))

class VGG(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(VGG, self).__init__()
        self.archtecture = architecture_config
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv_layers = self._create_conv_layers(self.archtecture)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc_layers = self._create_fc_layers(self.num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = self.fc_layers(torch.flatten(x, start_dim=1))
        out = F.softmax(x)
        return out

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
            elif type(x) == int:
                layers += [CNNBlock(in_channels, x, kernel_size=3, stride=1, padding=1)]
                in_channels = x

        return nn.Sequential(*layers)

    def _create_fc_layers(self, num_classes):

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.Linear(4096, 4096),
            nn.Linear(4096, num_classes)
        )

if __name__ == "__main__":
    image = torch.randn((1, 3, 224, 224))
    model = VGG()
    output = model(image)
    print(output.shape)
    print(model)