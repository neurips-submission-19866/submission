from __future__ import print_function
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset, ConcatDataset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import time

# python mnist.py & python mnist.py --softmax & python mnist.py --normalize & python mnist.py --softmax --normalize & python mnist.py --lenet & python mnist.py --softmax --lenet & python mnist.py --normalize --lenet & python mnist.py --softmax --normalize --lenet

model_name = "linear256"
USE_SOFTMAX = False # keep to false and use CLI params!
USE_LENET = False

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        if USE_LENET:
            # LeNet-5
            self.conv1 = nn.Conv2d(1, 6, 5, 1, padding=2)
            self.conv2 = nn.Conv2d(6, 16, 5, 1)
            self.fc1 = nn.Linear(400, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)
        else:
            self.fc1 = nn.Linear(784, 100)
            self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        if USE_LENET:
            x = self.conv1(x)
            x = F.leaky_relu(x, 0.1)
            x = F.max_pool2d(x, 2)
            x = self.conv2(x)
            x = F.leaky_relu(x, 0.1)
            x = F.max_pool2d(x, 2)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.leaky_relu(x, 0.1)
            x = self.fc2(x)
            x = F.leaky_relu(x, 0.1)
            x = self.fc3(x)
            if USE_SOFTMAX:
                x = F.log_softmax(x, dim=1)
            return x
        else:
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.leaky_relu(x, 0.1)
            x = self.fc2(x)
            return x



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        if not USE_SOFTMAX:
            target = F.one_hot(target, num_classes=10).float()

        optimizer.zero_grad()
        output = model(data)
        if USE_SOFTMAX:
            loss = F.nll_loss(output, target)
        else:
            loss = F.mse_loss(output, target)
        loss.backward()

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader, addition="", save_labels=True):
    model.eval()
    test_loss = 0
    correct = 0

    all_preds = []
    targets = []

    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

def main():
    # Sorry for using (and modifying!) global variables... I didn't have the time to build a proper torch setup here...
    global USE_SOFTMAX
    global USE_LENET
    global model_name
    global directory

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=90, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    
    # Own attributes
    parser.add_argument('--softmax', action='store_true', default=False,
                        help='')
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='')
    parser.add_argument('--lenet', action='store_true', default=False,
                        help='')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()


    # Swich between models
    if args.lenet == True:
        USE_LENET = True
        model_name = "lenet"

    # Set USE_SOFTMAX appropriately
    assert USE_SOFTMAX == False
    if args.softmax:
        USE_SOFTMAX = True
        model_name += "_softmax"
    else:
        model_name += "_regression"


    for n in [320, 3200]:
    # for n in [60000]:
        torch.manual_seed(args.seed)

        if use_cuda:
            device = torch.device("cuda")
        elif use_mps:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        # device = torch.device("cpu")

        train_kwargs = {'batch_size': args.batch_size}
        test_kwargs = {'batch_size': args.test_batch_size}
        if use_cuda:
            cuda_kwargs = {'num_workers': 1,
                        'pin_memory': True,
                        'shuffle': True}
            train_kwargs.update(cuda_kwargs)
            test_kwargs.update(cuda_kwargs)

        transform=transforms.Compose([
            transforms.ToTensor(),
            ])
        if args.normalize:
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

        dataset1 = datasets.MNIST('../data', train=True, download=True,
                        transform=transform)
        dataset2 = datasets.MNIST('../data', train=False,
                        transform=transform)
        dataset2 = ConcatDataset([dataset2] * 100)
        dataset1 = Subset(dataset1, list(range(n)))

        train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)


        # Also load Fashion MNIST
        dataset_fashion = datasets.FashionMNIST('../data', train=False,
                        transform=transform, download=True)
        fashion_loader = torch.utils.data.DataLoader(dataset_fashion, **test_kwargs)


        model = Net().to(device)
        optimizer = optim.SGD(model.parameters(), lr=1e-3)

        start_time = time.time()  # Start timer
        for epoch in tqdm(range(2)):
            train(args, model, device, train_loader, optimizer, epoch)
        elapsed_time = time.time() - start_time
        print(f"Time taken: {elapsed_time:.6f} seconds")
        print(device)
        print("")

        # # Eval
        # print("Number of Examples: ", n, "\n")

        start_time = time.time()  # Start timer
        test(model, device, test_loader, "_" + model_name + f"_{n}")
        elapsed_time = time.time() - start_time
        print(f"Time taken: {elapsed_time:.6f} seconds")
        print(device)

        # test(model, device, test_loader, "_" + model_name + f"_{n}")
        # test(model, device, fashion_loader, "2_" + model_name + f"_{n}", save_labels=False)

        if args.save_model:
            torch.save(model.state_dict(), "mnist_cnn.pt")


# MLP (2 epochs) CPU: 0.797961 seconds
# MLP (2 epochs) GPU: 0.688745 seconds

# LeNet (6 epochs) CPU: 3.3296376667 seconds
# LeNet (6 epochs) GPU: 2.320558 seconds

# LeNet Inference CPU: 17.553371 seconds
# LeNet Inference GPU: 15.933233 seconds


if __name__ == '__main__':
    main()