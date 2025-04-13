import torch
import torch.nn as nn
from torchinfo import summary

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CNN_LSTM(nn.Module):
    def __init__(self, backbone: str, hidden_size: int, num_layers_lstm: int, in_channels: int = 3, num_classes=2):
        super(CNN_LSTM, self).__init__()
        self.in_shape = 64

        if backbone == "resnet18":
            layers = [2,2,2] # layers = [2,2,2,2]
        elif backbone == "resnet34":
            layers = [3,4,3] # layers = [3,4,6,3]
        else:
            raise NotImplementedError
        
        self.first_conv = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.block1 = self._make_layer(ResidualBlock, 64, layers[0])
        self.block2 = self._make_layer(ResidualBlock, 128, layers[1], stride=2)
        self.block3 = self._make_layer(ResidualBlock, 256, layers[2], stride=2)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(input_size=256, hidden_size=hidden_size, num_layers=num_layers_lstm, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

        self.hidden = None
            
    def _make_layer(self, block, out_shape, layers, stride=1):
        downsample = None
        if stride != 1 or self.in_shape != out_shape:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_shape, out_shape, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_shape)
            )
        
        layer_list = []
        layer_list.append(block(self.in_shape, out_shape, stride, downsample))
        self.in_shape = out_shape
        for i in range(1, layers):
            layer_list.append(block(self.in_shape, out_shape))

        return nn.Sequential(*layer_list)

    def forward(self, x):
        hidden = None

        for t in range(x.size(1)):
            cur_frame = x[:,t,:,:,:].squeeze(1)
            out = self.first_conv(cur_frame)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.max_pool(out)

            out = self.block1(out)
            out = self.block2(out)
            out = self.block3(out)

            out = self.adaptive_avg_pool(out)
            out = torch.reshape(out, (out.shape[0], out.shape[1]*out.shape[2]*out.shape[3]))
            out, hidden = self.lstm(out, hidden)

        out = self.fc(out)
        out = torch.sigmoid(out)

        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        if self.downsample is not None:
            res = self.downsample(x)
        return self.relu(out+res)

if __name__ == '__main__':
    net = CNN_LSTM(backbone="resnet18", hidden_size=256, num_layers_lstm=1, num_classes=2).to(DEVICE)
    summary(net, input_size=(1,90,3,172,172))