import torch
import torch.nn as nn
import torch.nn.functional as F

architecture_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

class VGG(nn.Module):
    def __init__(self,
                 in_channels : int = 3,
                 num_classes : int = 10,
                 init_weights : bool = True,
                 ) -> None:
        super(VGG, self).__init__()
        self.archtecture = architecture_config
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv_layers = self._create_conv_layers(self.archtecture)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avgpool(x)
        out = self.fc_layers(torch.flatten(x, start_dim=1))
        return out


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization
                # mode fan_out : 'fan_out' preserves the magnitude of the variance of the weights in the backwards pass.
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                # initialize bias by 0
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, stride=1, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(),
                           ]
                in_channels = x

        return nn.Sequential(*layers)


if __name__ == "__main__":
    image = torch.randn((2, 3, 32, 32))
    model = VGG()
    output = model(image)
    print(output.shape)