import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

# https://www.163.com/dy/article/IGBSNTEU0519EA27.html
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, device):
        super(ResNetFeatureExtractor, self).__init__()
        self.device = device   
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224), 
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.to(device)

    def forward(self, img):
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        feature = self.model(img_tensor).squeeze()
        return feature
