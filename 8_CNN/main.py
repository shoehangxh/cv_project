import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import math
import torch.optim as optim

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ]
)
trainset = torchvision.datasets.CIFAR10(root="E:/dataset/cifar10/"
                                        ,train=True
                                        ,download=True
                                        ,transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)
testset = torchvision.datasets.CIFAR10(root="E:/dataset/cifar10/"
                                        ,train=False
                                        ,download=True
                                        ,transform=transform)
testloader = torch.utils.data.DataLoader(
    testset
    ,batch_size=4
    ,shuffle=True
    ,num_workers=0
)
cifar10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

cfg = {'VGG16':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}

class VGG(nn.Module):
    def __init__ (self, net_name):
        super().__init__()
        self.features = self.make_layers(cfg[net_name])
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] *m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
                m.bias.data.zero_()
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    def make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding = 1),
                           nn.BatchNorm2d(v),
                           nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

net = VGG('VGG16')
#net = net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.1
                       #, momentum=0.9
                       )

for epoch in range(5):
    train_loss = 0.0
    for batch_idx, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        inputs = inputs
        labels = labels
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if batch_idx % 2000 == 1999:
            print('[%d, %5d] loss : %.3f '% (epoch + 1, batch_idx + 1, train_loss / 2000))
            train_loss = 0.0
    print('saving epoch %d model ...' % (epoch + 1))
    state = {
        'net' : net.state_dict(),
        'epoch' : epoch + 1,
        }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/cifar10_epoch_%d.ckpt' % (epoch + 1))
    print('Finished training')


