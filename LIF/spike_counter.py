import tonic

dataset = tonic.datasets.NMNIST(save_to='./data', train=True)
events, target = dataset[0]
# tonic.utils.plot_event_grid(events)
import tonic.transforms as transforms

sensor_size = tonic.datasets.NMNIST.sensor_size

# Denoise removes isolated, one-off events
# time_window
frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000),
                                      transforms.ToFrame(sensor_size=sensor_size,
                                                         time_window=1000)
                                     ])

trainset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=True)
testset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=False)

from torch.utils.data import DataLoader
from tonic import DiskCachedDataset


cached_trainset = DiskCachedDataset(trainset, cache_path='./cache/nmnist/train')
cached_dataloader = DataLoader(cached_trainset)

batch_size = 128
trainloader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors())

def load_sample_batched():
    events, target = next(iter(cached_dataloader))

import torch
import torchvision

transform = tonic.transforms.Compose([torch.from_numpy,
                                      torchvision.transforms.RandomRotation([-10,10])])

cached_trainset = DiskCachedDataset(trainset, transform=transform, cache_path='./cache/nmnist/train')

# no augmentations for the testset
cached_testset = DiskCachedDataset(testset, cache_path='./cache/nmnist/test')

batch_size = 1
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np

dataset_length = len(cached_trainset)
indices = np.arange(0, dataset_length, 100)
sampler = SubsetRandomSampler(indices)

trainloader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=False, sampler=sampler)
# trainloader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=True)
print(len(trainloader))
testloader = DataLoader(cached_testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False))

import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import spikeplot as splt
from snntorch import utils
import torch.nn as nn
import torch

import os

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# neuron and simulation parameters
spike_grad = surrogate.atan()
beta = 0.5

#  Initialize Network
net = nn.Sequential(nn.Conv2d(2, 12, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Conv2d(12, 32, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Flatten(),
                    nn.Linear(32*5*5, 10),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                    ).to(device)

# this time, we won't return membrane as we don't need it

def forward_pass(net, data):
  spk_rec = []
  utils.reset(net)  # resets hidden states for all LIF neurons in net

  for step in range(data.size(0)):  # data.size(0) = number of time steps
      spk_out, mem_out = net(data[step])
      spk_rec.append(spk_out)

  return torch.stack(spk_rec)

optimizer = torch.optim.Adam(net.parameters(), lr=2e-2, betas=(0.9, 0.999))
loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

from tqdm import tqdm
import numpy as np
import os
num_epochs = 10
num_iters = 10

loss_hist = []
acc_hist = []

use_pretrained = False

if os.path.exists('./model/nmnist.pth') and use_pretrained:
  net.load_state_dict(torch.load('./model/nmnist.pth'))
  print('Model loaded')
else:
  # training loop
  for epoch in range(num_epochs):
      tqdm.write(f"Epoch {epoch + 1}\n-------------------------------")
      spk_list = []
      target_list = []
      for i, (data, targets) in tqdm(enumerate(iter(trainloader))):
          data = data.to(device)
          targets = targets.to(device)

          net.train()
          spk_rec = forward_pass(net, data)
          loss_val = loss_fn(spk_rec, targets)

          # Gradient calculation + weight update
          optimizer.zero_grad()
          loss_val.backward()
          optimizer.step()

          # Store loss history for future plotting
          loss_hist.append(loss_val.item())
          # acc = SF.accuracy_rate(spk_rec, targets)
          _, idx = spk_rec.sum(dim=0).max(1)
          spk_list.extend(idx)
          target_list.extend(targets)
          # if i == num_iters:
          #   break
      spk_list = torch.stack(spk_list).tolist()
      target_list = torch.stack(target_list).tolist()
      accuracy = np.mean((spk_list == target_list).detach().cpu().numpy())
      acc_hist.append(accuracy)
      # print(f"Accuracy: {acc * 100:.2f}%\n")
      tqdm.write(f"Accuracy: {accuracy * 100:.2f}%\n")
          # training loop breaks after 50 iterations
          # if i == num_iters:
          #   break

torch.save(net.state_dict(), './model/nmnist.pth')

# acc per class
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

pred = spk_list
target = target_list

cm = confusion_matrix(target, pred)
acc = accuracy_score(target, pred)

print(f'Accuracy: {acc}')

for i in range(10):
  print(f'Accuracy for class {i}: {cm[i,i]/cm[i,:].sum()}')

plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Accuracy: {acc}')
plt.savefig('./figures/nmnist_confusion_matrix.png')
plt.show()

import matplotlib.pyplot as plt

# Plot Loss
fig = plt.figure(facecolor="w")
plt.plot(acc_hist)
plt.title("Train Set Accuracy")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.savefig('./figures/nmnist_accuracy.png')
plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_dict = {}
import torch
for data, target in tqdm(trainloader):
  if target.item() in data_dict:
    continue
  data = torch.tensor(data)
  data = data.to(device)
  data_dict[target.item()] = data
#   break
  if len(data_dict) == 10:
    break
  
for k, v in data_dict.items():
    fig, ax = plt.subplots(facecolor='w', figsize=(12, 7))
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    print(f"The target label is: {labels[k]}")
    data = v.to(device)
    spk_rec = forward_pass(net, data)
    # print(spk_rec.shape)
    # 创建一个空白图像，用于绘制动画的每一帧
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # 创建一个可写的副本
    frame_writable = frame.copy()

    # 获取脉冲计数数据
    spike_counts = spk_rec[:, idx].detach().cpu()
    df = pd.DataFrame(spike_counts.numpy())
    df.to_csv(f'spike_count_{k}.csv', index=False)
    # 创建颜色映射
    cmap = plt.get_cmap('viridis')

    # 创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'spike_count_{k}.mp4', fourcc, 10.0, (frame.shape[1], frame.shape[0]))

    # 绘制动画的每一帧
    for t in range(len(spike_counts)):
        # 清空图像
        frame_writable.fill(0)

        # 绘制条形图
        for i, count in enumerate(spike_counts[t]):
            color = cmap(i / len(labels))[:3]
            color = list((np.array(color) * 255).astype(int))
            # print(color)
            color = [0, 255, 0]
            cv2.rectangle(frame_writable, (i * 50, 0), ((i + 1) * 50, int(count * 100)),
                        color, -1)
            cv2.putText(frame_writable, labels[i], (i * 50 + 10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2, cv2.LINE_AA)

        # 将当前帧写入视频
        out.write(frame_writable)

    # 释放VideoWriter对象
    out.release()