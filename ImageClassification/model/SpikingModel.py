import torch.autograd
import torch.nn as nn
from snntorch import surrogate

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class MLP(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(MLP, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.fc = nn.Linear(in_features=self.input_shape,
                            out_features=self.output_shape)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.ReLU(x)
        return x


class hybrid_neuron(nn.Module):
    def __init__(self, input_shape, output_shape, T, batch_size):
        super(hybrid_neuron, self).__init__()
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.MLP = MLP(input_shape=self.input_shape,
                       output_shape=self.output_shape).float().to(device)
        self.T = T  # num steps
        self.sigmoid = surrogate.fast_sigmoid()
        self.alpha = torch.autograd.Variable(torch.tensor(0.),
                                             requires_grad=True).to(device)  # Alpha, Eta, Beta
        self.eta = torch.autograd.Variable(torch.tensor(0.),
                                           requires_grad=True).to(device)
        self.beta = torch.autograd.Variable(torch.tensor(0.),
                                            requires_grad=True).to(device)
        self.tao_u = torch.tensor(5).to(device)  # membrane time constant
        self.tao_w = torch.tensor(1).to(device)  # synaptic constant
        self.dt = torch.tensor(1).to(device)
        self.s = torch.zeros(self.output_shape, self.batch_size).to(device)
        self.u = torch.zeros(self.output_shape, self.batch_size).to(device)
        self.k = self.dt / self.tao_u
        self.V_th = torch.tensor(0.2).to(device)
        self.P = torch.autograd.Variable(torch.zeros((self.output_shape, self.input_shape)),
                                         requires_grad=True).to(device)

    def forward(self, s_batch):
        spk = []
        for m in range(self.T):
            # for s in s_batch:
            s = s_batch
            decay_1 = torch.tensor(-m) / self.tao_w
            self.u = torch.mul((1 - self.s) * (1 - self.k), self.u) + \
                     self.k * ((self.MLP(s) * decay_1.exp()).T + self.alpha * torch.matmul(self.P, s.T))
            self.P = self.P * (-self.dt / self.tao_w).exp() + \
                     self.eta * torch.matmul(self.sigmoid(self.u) + self.beta,
                                             s)
            self.s = self.sigmoid(self.u - self.V_th)
            spk.append(self.s.T)
            self.reset()
        # spk_batch = []
        # for s in s_batch:
        #     spk = []
        #     for m in range(self.T):
        #         decay_1 = torch.tensor(-m) / self.tao_w
        #         self.u = (1 - self.s) * (1 - self.k) * self.u + \
        #                  self.k * (self.MLP(s) * decay_1.exp() + self.alpha * torch.matmul(self.P, s)).sum()
        #         self.P = self.P * (-self.dt / self.tao_w).exp() + \
        #                  self.eta * torch.matmul(torch.stack([self.sigmoid(self.u) + self.beta]).T,
        #                                          torch.stack([s]))
        #         self.s = self.sigmoid(self.u - self.V_th)
        #         spk.append(self.s)
        #     self.reset()
        #     spk = torch.stack(spk)
        #     spk_batch.append(spk)
        return torch.stack(spk)

    def reset(self):
        self.s = torch.zeros(self.output_shape, self.batch_size).to(device)
        self.u = torch.zeros(self.output_shape, self.batch_size).to(device)


if __name__ == '__main__':
    device = 'cuda'
    h_n = hybrid_neuron(input_shape=10,
                        output_shape=8,
                        T=5,
                        batch_size=2).float().to(device)
    input_feature = torch.tensor([[0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
                                  [0, 0, 0, 1, 0, 0, 1, 0, 1, 0]]).float().to(device)
    output = h_n(input_feature)
    output.sum().backward()
    # time steps; batch_size; class nums
