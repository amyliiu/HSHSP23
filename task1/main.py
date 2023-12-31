#Training CIFAR10 with Python

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

import pickle
import numpy as np

from models.basic import Net

# Define hyperparameters
batch_size = 64
learning_rate = 0.001
momentum = 0.9
num_epochs = 20

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load and prepare the CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)

# Create data loaders
trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=0)

# Define class labels
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)


def train():
    # xvals = np.empty(1)
    # yvals = np.empty(1)
    # plt.xlabel( 'num-epoch')
    # plt.ylabel( 'running loss')

    global num_epochs

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 4000 == 1999:
                print(f'[{epoch+1}, {i+1}] loss: {running_loss/2000:.3f}')
                running_loss = 0.0
        
        #np.append(xvals, running_loss)
        #np.append( yvals, running_loss)
        print( "Epoch: " + str(epoch+1) + "\nRunning Loss: " + str(running_loss ))
        #plt.plot(xvals, yvals)
        #plt.show()

    PATH = '/Users/amyliu/Documents/HSHSP23/task1/cifar_net.pth'
    torch.save(net.state_dict(), PATH)
    print('Saved.')

def test(net):
    global testloader, classes
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct / total:.2f}%')



train()


PATH = '/Users/amyliu/Documents/HSHSP23/task1/cifar_net.pth'
model = Net()
net.load_state_dict(torch.load(PATH))

model.eval()
print("Model loaded")

test(model)

