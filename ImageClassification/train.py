import snntorch.functional as SF
import torch
from tqdm import trange

from model.SpikingModel import hybrid_neuron
from spiking.SpikeEncoder import SimpleEncoder

batch_size = 100
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

net = hybrid_neuron(input_shape=28 * 28,
                    output_shape=10,
                    T=3,
                    batch_size=batch_size).float().to(device)
optimizer = torch.optim.Adam(list(net.parameters()), lr=2e-3, betas=(0.9, 0.999))
loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
loss_hist = []
acc_hist = []
num_epochs = 1
# train loop
for epoch in trange(num_epochs):
    for batch_idx, (img, target) in enumerate(train_loader):
        net.train()
        img = SimpleEncoder(img_batch=img)
        img, target = img.to(device), target.to(device)

        spk_rec = net(img)
        loss_val = loss_fn(spk_rec, target)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward(retain_graph=True)
        # loss_val.backward()
        optimizer.step()
        # Store loss history for future plotting
        loss_hist.append(loss_val.item())
        """
        This step is important.
        """
        loss_val.zero_()
        # print every 25 iterations
        if batch_idx % 25 == 0:
            # if 1:
            net.eval()
            print(f"Epoch {epoch}, Iteration {batch_idx} \nTrain Loss: {loss_val.item():.2f}")

            # check accuracy on a single batch
            acc = SF.accuracy_rate(spk_rec, target)
            acc_hist.append(acc)
            print(f"Accuracy: {acc * 100:.2f}%\n")
        pass
