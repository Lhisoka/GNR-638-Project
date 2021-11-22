import time
import torch
import torchvision
import torchvision.transforms as transforms

def main():
    device = 'cpu'

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])

    batch_size = 50

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = [str(x) for x in range(10)]
    classes

    import matplotlib.pyplot as plt
    import numpy as np

    # functions to show an image

    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()


    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

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


    class LeNet(nn.Module):
        def __init__(self, k_type='', k_pos=0):

            super(LeNet,self).__init__()
            self.k_pos=k_pos
            self.kerv1 = Kerv2d(in_channels=1,out_channels=6,kernel_size=(5,5),padding=0,stride=1, kernel_type=k_type)
            self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=(5,5),padding=0,stride=1)
            self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16,kernel_size = (5,5),padding=0,stride=1)
            self.kerv2 = Kerv2d(in_channels = 6, out_channels = 16,kernel_size = (5,5),padding=0,stride=1, kernel_type=k_type)
            self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 120,kernel_size = (4,4),padding=0,stride=1)
            self.kerv3 = Kerv2d(in_channels = 16, out_channels = 120,kernel_size = (4,4),padding=0,stride=1, kernel_type=k_type)
            self.L1 = nn.Linear(120,84)
            self.L2 = nn.Linear(84,10)
            self.avgpool = nn.AvgPool2d(kernel_size = 2, stride = 2)
            self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
            self.act = nn.Tanh()
            
        def forward(self,x):
            if self.k_pos == 1:
                x = self.kerv1(x)
                x = self.avgpool(x)
            else:
                x = self.conv1(x)
                x = self.maxpool(x)
            if self.k_pos == 2:
                x = self.kerv2(x)
                x =  self.avgpool(x)
            else:
                x = self.conv2(x)
                x = self.maxpool(x)
            if self.k_pos == 3:
                self.kerv3(x)
            else:
                x = self.conv3(x)
            
            x = x.view(x.size()[0], -1)
            x = self.L1(x)
            x = self.act(x)
            x = self.L2(x)
            return x


    def train(network,num_epochs, PATH='', lr=0.003):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(network.parameters(), lr=lr, momentum=0.9, dampening=0.1)

        running_losses=[]
        start_time = time.time()
        for epoch in range(num_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            count=0
            for i, data in enumerate(trainloader, 0):
                count+=1
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = network(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 50 == 49:    # print every 250 mini-batches, store running_loss every 50 mini batches.
                    if i%250 == 249:
                        print('[%d, %5d] loss: %.7f' %
                            (epoch + 1, i + 1, running_loss / 50))
                    running_losses.append(running_loss/50)
                    running_loss = 0.0
        end_time = time.time()
        print('Finished Training, Execution Time: {} '.format(end_time-start_time))
        if PATH != '':
            torch.save(network.state_dict(), PATH)
        return (running_losses,end_time-start_time)

    def load(network, PATH): 
        network.load_state_dict(torch.load(PATH))
        network = network.to(device)
        images = images.to(device)
        return network

    def test_acc(network):
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                # calculate outputs by running images through the network 
                outputs = network(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %f %%' % (
            100 * correct / total))

    def generate_results():
        kernel_types=['polynomial']
        kernel_positions=[1,2]
        train_dic = {}
        test_dic = {}
        network = LeNet().to(device)
        # print('training conv')
        PATH = './mnist_conv'
        epochs=5
        train_dic['conv'] = train(network, num_epochs=epochs,PATH=PATH)
        test_dic['conv'] = test_acc(network)
        for k_type in kernel_types:
            for k_pos in kernel_positions:
                network = LeNet(k_type,k_pos).to(device)
                PATH = './mnist_{}_{}'.format(k_type,k_pos)
                print(k_type,k_pos,'training')
                train_dic[k_type+str(k_pos)] = train(network,num_epochs=epochs,PATH=PATH, lr=0.003)
                test_dic[k_type + str(k_pos)] = test_acc(network)
        return train_dic,test_dic

    
    train_dic,test_dic = generate_results()
    plt.plot(train_dic['conv'][0],label='conv')
    plt.plot(train_dic['polynomial1'][0], label='poly-linear-linear')
    plt.plot(train_dic['polynomial2'][0],label='linear-poly-linear')
    plt.legend()
    plt.xlabel('train time')
    plt.ylabel('loss value')
    plt.title('speed of training on MNIST')
    plt.show()


if __name__ == '__main__':
    main()