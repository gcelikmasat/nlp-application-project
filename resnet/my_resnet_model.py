from torch import nn
from torchvision.models import resnet152


class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet152(pretrained=True)
        # Remove the last fully connected layer (fc layer)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.resnet(x)  # Shape will be (batch_size, 2048, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 2048)
        return x
