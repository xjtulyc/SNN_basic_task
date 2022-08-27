import snntorch as snn
import snntorch.functional as SF
import torch
import torch.nn as nn

batch_size = 128
data_path = 'data/mnist'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define a transform
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

# Create DataLoaders
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

from snntorch import surrogate

beta = 0.9  # neuron decay rate
spike_grad = surrogate.fast_sigmoid()


# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        num_inputs = 784
        num_hidden = 300
        num_outputs = 10
        spike_grad = surrogate.fast_sigmoid()

        # global decay rate for all leaky neurons in layer 1
        beta1 = 0.9
        # independent decay rate for each leaky neuron in layer 2: [0, 1)
        beta2 = torch.rand((num_outputs), dtype=torch.float)  # .to(device)

        # Init layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta1, spike_grad=spike_grad, learn_beta=True)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta2, spike_grad=spike_grad, learn_beta=True)

    def forward(self, x):
        # reset hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x.flatten(1))
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec), torch.stack(mem2_rec)


# Load the network onto CUDA if available
net = Net().to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=2e-3, betas=(0.9, 0.999))
loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

num_epochs = 1
num_steps = 100  # run for 25 time steps

loss_hist = []
acc_hist = []

# training loop
for epoch in range(num_epochs):
    for i, (data, targets) in enumerate(iter(train_loader)):
        data = data.to(device)
        targets = targets.to(device)

        net.train()
        spk_rec, _ = net(data)
        loss_val = loss_fn(spk_rec, targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        # print every 25 iterations
        if i % 25 == 0:
            net.eval()
            print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")

            # check accuracy on a single batch
            acc = SF.accuracy_rate(spk_rec, targets)
            acc_hist.append(acc)
            print(f"Accuracy: {acc * 100:.2f}%\n")

        # uncomment for faster termination
        # if i == 150:
        #     break

print(f"Trained decay rate of the first layer: {net.lif1.beta:.3f}\n")

print(f"Trained decay rates of the second layer: {net.lif2.beta}")

def _test_accuracy(data_loader, net, num_steps):
  with torch.no_grad():
    total = 0
    acc = 0
    net.eval()

    data_loader = iter(data_loader)
    for data, targets in data_loader:
      data = data.to(device)
      targets = targets.to(device)
      spk_rec, _ = net(data)

      acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
      total += spk_rec.size(1)

  return acc/total

print(f"Test set accuracy: {_test_accuracy(test_loader, net, num_steps)*100:.3f}%")