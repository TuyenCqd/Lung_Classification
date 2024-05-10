import torch
import torch.nn as nn

class VGGNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = self._make_block_2conv(3, 64)
        self.conv2 = self._make_block_2conv(64, 128)
        self.conv3 = self._make_block_3conv(128, 256)
        self.conv4 = self._make_block_3conv(256, 512)
        self.conv5 = self._make_block_3conv(512, 512)

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        soft_max = nn.Softmax(dim=1)
        x = soft_max(x)
        return x
    def _make_block_2conv(self, in_channels, out_channels, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def _make_block_3conv(self, in_channels, out_channels, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

if __name__ == '__main__':
    fake_data = torch.rand(8, 3, 224, 224)
    # print(fake_data)
    # model = VGGNet(num_classes=10)
    model = VGGNet(num_classes=2)
    predection = model(fake_data)
    print(predection.shape)