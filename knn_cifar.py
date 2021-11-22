# -*- coding: utf-8 -*-
"""CNN CIFAR.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Xt1QE5nnUzhoS2tT_mu3rm_Yj19NO9is
"""

import time
import torch
import torchvision
import torchvision.transforms as transforms


def main():
    device = 'cpu'

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # classes = [str(x) for x in range(10)]

    import matplotlib.pyplot as plt
    import numpy as np

    class Kerv2d(nn.Conv2d):
        def __init__(self, in_channels, out_channels, kernel_size, 
                stride=1, padding=0, dilation=1, groups=1, bias=True,
                kernel_type='polynomial', balance=1, power=3, gamma=1):

            super(Kerv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
            self.kernel_type = kernel_type
            self.balance, self.power, self.gamma = balance, power, gamma

        def forward(self, input):
            minibatch, in_channels, input_width, input_hight = input.size()
            assert(in_channels == self.in_channels)
            input_unfold = F.unfold(input, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
            input_unfold = input_unfold.view(minibatch, 1, self.kernel_size[0]*self.kernel_size[1]*self.in_channels, -1)
            weight_flat  = self.weight.view(self.out_channels, -1, 1)
            output_width = (input_width - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0] + 1
            output_hight = (input_hight - self.kernel_size[1] + 2 * self.padding[1]) // self.stride[1] + 1

            if self.kernel_type == 'linear': output = (input_unfold * weight_flat).sum(dim=2)
            elif self.kernel_type == 'polynomial': output = ((input_unfold * weight_flat).sum(dim=2) + self.balance)**self.power
            elif self.kernel_type == 'gaussian': output = (-self.gamma*((input_unfold - weight_flat)**2).sum(dim=2)).exp() + 0
            else: raise NotImplementedError(self.kernel_type+' kervolution not implemented')

            if self.bias is not None: output += self.bias.view(self.out_channels, -1)
            return output.view(minibatch, self.out_channels, output_width, output_hight)

    import torch.nn as nn
    import torch.nn.functional as F


    class LeNet(nn.Module):
        def __init__(self):
            super(LeNet,self).__init__()
            self.conv1 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=(5,5),padding=0,stride=1)
            self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16,kernel_size = (5,5),padding=0,stride=1)
            self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 30,kernel_size = (4,4),padding=0,stride=1)
            self.L1 = nn.Linear(120,84)
            self.L2 = nn.Linear(84,10)
            self.pool = nn.AvgPool2d(kernel_size = 2, stride = 2)
            self.act = nn.Tanh()
            
        def forward(self,x):
            x = self.conv1(x)
            x = self.act(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = self.act(x)
            x = self.pool(x)
            x = self.conv3(x)
            x = self.act(x)    
            x = x.view(x.size()[0], -1)
            x = self.L1(x)
            x = self.act(x)
            x = self.L2(x)
            return x

    # net = LeNet().to(device)
    # net.load_state_dict(torch.load(PATH))

    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_planes, planes, stride=1):
            super(BasicBlock, self).__init__()
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                            kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out


    class Bottleneck(nn.Module):
        expansion = 4

        def __init__(self, in_planes, planes, stride=1):
            super(Bottleneck, self).__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, self.expansion *
                                planes, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(self.expansion*planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                            kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out


    class ResNet(nn.Module):
        def __init__(self, block, num_blocks, num_classes=10):
            super(ResNet, self).__init__()
            self.in_planes = 64

            self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                                stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            self.linear = nn.Linear(512*block.expansion, num_classes)

        def _make_layer(self, block, planes, num_blocks, stride):
            strides = [stride] + [1]*(num_blocks-1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out


    def ResNet18():
        return ResNet(BasicBlock, [2, 2, 2, 2])

    net = ResNet18().to(device)

    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.003, momentum=0.9, dampening=0.1)


    for epoch in range(0):  # loop over the dataset multiple times
        start_time = time.time()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        print(f'Epoch: {epoch+1}, Time elapsed: {time.time() - start_time}')
    print('Finished Training')

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    net = ResNet18()
    net.load_state_dict(torch.load(PATH))
    net = net.to(device)

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network 
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %f %%' % (
        100 * correct / total))

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)    
            outputs = net(images)    
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname, 
                                                    accuracy))


if __name__ == '__main__':
    main()