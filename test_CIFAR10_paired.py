# Import various libraries
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Callable, Optional, Tuple
import numpy as np
import torch.multiprocessing as mp
import time
import random
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import opacus
from collections import deque
import os
import copy
import warnings
import pickle
import json
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ─── Hyperparameters ──────────────────────────────────────────────────────────
# Must stay consistent with train_CIFAR10_improved.py
N          = 5       # Number of clients (match training)
batch_size = 200
episodes   = int(os.environ.get('N_TRIALS', 10))   # paired trials (was 1)
CR_Total   = 25
rho_min    = 20.0
rho_max    = 40.0
epoch_local = 4
DRL_steps  = 1
sigma_max  = 20.0
sigma_min  = 1.0
rho_used_min = 2 / sigma_max**2 * epoch_local
rho_used_max = 2 / sigma_min**2 * epoch_local

# Episode of the best saved model to load for evaluation.
# Adjust to whichever episode number agent.save() last wrote.
LOAD_EPISODE = 181   # Set to an int to use agent.load(); None → use checkpoint

CHECKPOINT_PATH = './DRL_Nets/checkpoint_CIFAR10.pth'

# ─── Conversion helpers (must match train_CIFAR10_improved.py) ────────────────
def rho2sigma(rho, el, bs):
    return np.sqrt(2 * el / rho)

def sigma2rho(sigma, el, bs):
    return 2 * el / (sigma**2)


# ─── Dataset ──────────────────────────────────────────────────────────────────
class CUDACIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 pre_transform: Callable[..., Any] = None,
                 transform: Callable[..., Any] = None,
                 target_transform: Callable[..., Any] = None,
                 download: bool = False,
                 device: str = 'cuda:0') -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.data = self.data.astype('float32')
        for i in range(len(self)):
            self.data[i] = pre_transform(self.data[i] / 255.0).numpy().transpose((1, 2, 0))
            self.targets[i] = torch.Tensor([self.targets[i]]).squeeze_().long()
            self.targets[i] = self.targets[i].cuda(device=device)
        self.data = torch.Tensor(self.data).permute((0, 3, 1, 2))
        self.data = self.data.cuda(device=device)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class CUDACIFAR10_Test(torchvision.datasets.CIFAR10):
    """Separate class for the held-out test split so train=False is explicit."""
    def __init__(self,
                 root: str,
                 train: bool = False,
                 pre_transform: Callable[..., Any] = None,
                 transform: Callable[..., Any] = None,
                 target_transform: Callable[..., Any] = None,
                 download: bool = False,
                 device: str = 'cuda:0') -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.data = self.data.astype('float32')
        for i in range(len(self)):
            self.data[i] = pre_transform(self.data[i] / 255.0).numpy().transpose((1, 2, 0))
            self.targets[i] = torch.Tensor([self.targets[i]]).squeeze_().long()
            self.targets[i] = self.targets[i].cuda(device=device)
        self.data = torch.Tensor(self.data).permute((0, 3, 1, 2))
        self.data = self.data.cuda(device=device)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


# ─── Models ───────────────────────────────────────────────────────────────────
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16 * 5 * 5, 120)
        self.fc2   = nn.Linear(120, 64)
        self.fc3   = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, dim_state, dim_action):
        super().__init__()
        self.fc1 = nn.Linear(dim_state, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, dim_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x


class Critic(nn.Module):
    def __init__(self, dim_state, dim_action):
        super().__init__()
        self.fc1 = nn.Linear(dim_state + dim_action, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ─── Agent ────────────────────────────────────────────────────────────────────
class Agent():
    def __init__(self,
                 n_agents,
                 dim_state,
                 dim_action,
                 lr_actor=0.00002,
                 lr_critic=0.0002,
                 lr_actor_target=0.005,
                 lr_critic_target=0.005,
                 buffer_size=5000,
                 minimal_size=512,
                 gamma=0.998,
                 explore=1.0,
                 explore_decay=0.9996,
                 explore_decay_interval=1000,
                 bs=256,
                 train_interval=1,
                 device='cuda:0') -> None:

        self.device = device
        self.n_agents = n_agents
        self.dim_state = dim_state
        self.dim_action = dim_action

        self.actor        = Actor(dim_state=self.dim_state, dim_action=self.dim_action).cuda(device=device)
        self.actor_target = Actor(dim_state=self.dim_state, dim_action=self.dim_action).cuda(device=device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic        = Critic(dim_state=self.dim_state, dim_action=self.dim_action).cuda(device=device)
        self.critic_target = Critic(dim_state=self.dim_state, dim_action=self.dim_action).cuda(device=device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.lr_actor_target  = lr_actor_target
        self.lr_critic_target = lr_critic_target
        self.bs               = bs
        self.criterion_critic = nn.MSELoss()
        self.optimizer_actor  = optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.minimal_size = minimal_size
        self.rep_buffer = {
            'state':      [deque(maxlen=buffer_size) for _ in range(self.n_agents)],
            'action':     [deque(maxlen=buffer_size) for _ in range(self.n_agents)],
            'reward':     deque(maxlen=buffer_size),
            'next_state': [deque(maxlen=buffer_size) for _ in range(self.n_agents)],
            'done':       deque(maxlen=buffer_size),
        }

        self.gamma                 = torch.Tensor([gamma]).cuda(device=device)
        self.explore               = explore
        self.explore_decay         = explore_decay
        self.explore_decay_interval = explore_decay_interval
        self.step                  = 1
        self.train_step            = 1
        self.train_interval        = train_interval

    # Deterministic action for evaluation (no exploration noise)
    def take_action(self, state):
        with torch.no_grad():
            action = self.actor(torch.Tensor(state).cuda(device=self.device)).cpu().numpy()
            return action

    # Load best model by episode number
    def load(self, episode):
        self.actor.load_state_dict(
            torch.load('./DRL_Nets/actor_CIFAR10_' + str(episode) + '.pth', map_location=self.device))
        self.actor_target.load_state_dict(
            torch.load('./DRL_Nets/actor_target_CIFAR10_' + str(episode) + '.pth', map_location=self.device))
        self.critic.load_state_dict(
            torch.load('./DRL_Nets/critic_CIFAR10_' + str(episode) + '.pth', map_location=self.device))
        self.critic_target.load_state_dict(
            torch.load('./DRL_Nets/critic_target_CIFAR10_' + str(episode) + '.pth', map_location=self.device))

    # Load full training checkpoint
    def load_checkpoint(self, path=CHECKPOINT_PATH):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic'])
        buf    = checkpoint['rep_buffer']
        maxlen = self.rep_buffer['state'][0].maxlen
        self.rep_buffer['state']      = [deque(q, maxlen=maxlen) for q in buf['state']]
        self.rep_buffer['action']     = [deque(q, maxlen=maxlen) for q in buf['action']]
        self.rep_buffer['reward']     = deque(buf['reward'],     maxlen=maxlen)
        self.rep_buffer['next_state'] = [deque(q, maxlen=maxlen) for q in buf['next_state']]
        self.rep_buffer['done']       = deque(buf['done'],       maxlen=maxlen)
        self.explore    = checkpoint['explore']
        self.step       = checkpoint['step']
        self.train_step = checkpoint['train_step']
        print(f"[Checkpoint] Loaded from episode {checkpoint['episode']}")
        return checkpoint['episode'], checkpoint['training_state']


# ─── Training process (client subprocess) ─────────────────────────────────────
class Training(mp.Process):
    def __init__(self, C=6, bs=batch_size, lr=0.1, el=epoch_local,
                 ins=mp.Queue(), outs=mp.Queue(), device='cuda:0') -> None:
        mp.Process.__init__(self)
        self.device = device
        self.C      = C
        self.bs     = bs
        self.el     = el
        self.lr     = lr
        self.trainloader = None
        self.trset       = None
        self.ins    = ins
        self.outs   = outs

    def run(self):
        self.trset = self.ins.get()
        self.trainloader = torch.utils.data.DataLoader(
            self.trset, batch_size=self.bs, shuffle=True, num_workers=0, pin_memory=False)
        self.net = Net()
        pe = opacus.PrivacyEngine()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr)
        self.net, self.optimizer, _ = pe.make_private(
            module=self.net,
            data_loader=self.trainloader,
            optimizer=self.optimizer,
            noise_multiplier=1.0,
            max_grad_norm=self.C
        )
        self.net.cuda(device=self.device)
        print('Dataset loaded successfully on', self.device)
        self.outs.put(1)

        while True:
            mode, net, sigma = self.ins.get()
            if mode == 0:   # Terminate
                break
            elif mode == 1: # Train
                self.net.load_state_dict(net.state_dict())
                self.optimizer.noise_multiplier = sigma
                for epoch in range(self.el):
                    for i, data in enumerate(self.trainloader):
                        self.optimizer.zero_grad()
                        inputs, labels = data
                        outputs = self.net(inputs)
                        loss    = self.criterion(outputs, labels)
                        loss.backward()
                        self.optimizer.step()
                self.outs.put(self.net)
            else:           # Test on local data
                self.net.load_state_dict(net.state_dict())
                with torch.no_grad():
                    correct  = 0
                    total    = 0
                    avg_loss = 0
                    count    = 0
                    for i, data in enumerate(self.trainloader):
                        inputs, labels = data
                        outputs        = self.net(inputs)
                        loss           = self.criterion(outputs, labels)
                        _, predicted   = torch.max(outputs.data, 1)
                        total    += labels.size(0)
                        correct  += (predicted == labels).sum().item()
                        avg_loss  = avg_loss * (count / (count + 1)) + loss.item() / (count + 1)
                        count    += 1
                    acc = correct / total
                    self.outs.put([acc, avg_loss])


# ─── Client ───────────────────────────────────────────────────────────────────
class Client():
    def __init__(self, id=-1, C=6, rho_total=20.0, bs=batch_size, trset=None,
                 lr=0.1, el=epoch_local, hl=5, agent=None, device='cuda:0') -> None:
        self.id        = id
        self.rho       = rho_total
        self.rho_total = rho_total
        self.C         = C
        self.bs        = bs
        self.trset     = trset
        self.lr        = lr
        self.el        = el
        self.hl        = hl
        self.sigma     = sigma_max
        self.rho_used  = 2 * self.el / (self.sigma**2)
        self.local_acc = 0.1
        self.state     = None
        self.action    = None
        self.reward    = None
        self.next_state = None
        self.done      = None
        self.real_action = None
        self.device    = device
        self.agent     = agent

        self.ins  = mp.Queue(1)
        self.outs = mp.Queue(1)
        self.process = Training(C=self.C, bs=self.bs, lr=self.lr, el=self.el,
                                ins=self.ins, outs=self.outs, device=device)
        self.process.start()
        self.ins.put(self.trset)
        self.outs.get()

    def reset(self, rho_total):
        self.rho       = rho_total
        self.rho_total = rho_total
        self.action    = None
        self.sigma     = sigma_max
        self.local_acc = 0.1

    def train(self, net, sigma):
        self.sigma   = sigma
        rho_per_epoch = 2 / (sigma**2)
        self.rho_used = self.el * rho_per_epoch
        self.rho     -= self.rho_used
        self.ins.put([1, net, sigma])

    def test(self, net):
        self.ins.put([2, net, 0])

    def terminate_process(self):
        self.ins.put([0, 0, 0])


# ─── Server ───────────────────────────────────────────────────────────────────
class Server():
    def __init__(self, net, device='cuda:0') -> None:
        self.device = device
        self.net    = Net()
        self.net.load_state_dict(net.state_dict())
        dummyOptimizer = optim.SGD(self.net.parameters(), lr=0)
        dummySet       = [[0, 0]]
        dummyLoader    = torch.utils.data.DataLoader(dummySet)
        privacyEngine  = opacus.PrivacyEngine()
        self.net, _, _ = privacyEngine.make_private(
            module=self.net, optimizer=dummyOptimizer,
            data_loader=dummyLoader, noise_multiplier=4, max_grad_norm=4)
        self.dummy_net = copy.deepcopy(self.net)
        self.dummy_net.cuda(device=self.device)
        self.net.cuda(device=self.device)

    def reset(self, net):
        self.net = Net()
        self.net.load_state_dict(net.state_dict())
        dummyOptimizer = optim.SGD(self.net.parameters(), lr=0)
        dummySet       = [[0, 0]]
        dummyLoader    = torch.utils.data.DataLoader(dummySet)
        privacyEngine  = opacus.PrivacyEngine()
        self.net, _, _ = privacyEngine.make_private(
            module=self.net, optimizer=dummyOptimizer,
            data_loader=dummyLoader, noise_multiplier=4, max_grad_norm=4)
        self.net.cuda(device=self.device)

    def aggregate(self, net_list):
        count = 0
        for net in net_list:
            self.dummy_net.load_state_dict(net.state_dict())
            for name, p in self.net.named_parameters():
                p.data = p.data * (count / (count + 1)) + self.dummy_net.state_dict()[name].data / (count + 1)
            count += 1


# ─── Held-out test set evaluation ─────────────────────────────────────────────
def test(net, testloader, device):
    """Evaluate net on the CIFAR-10 test split; returns accuracy in [0,1]."""
    correct = 0
    total   = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.cuda(device=device)
            labels = labels.cuda(device=device)
            outputs        = net(images)
            _, predicted   = torch.max(outputs.data, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


# ─── Budget-projection helper (identical to training file) ────────────────────
def project_sigmas(sigmas, rho_useds, clients):
    """Apply the affine action-mapping module from the training script."""
    rho_used_sums = [sum(rho_useds[i]) for i in range(N)]
    a = 1.0
    b = 0.0
    for i, client in enumerate(clients):
        if abs(max(sigmas[i]) - min(sigmas[i])) < 0.01:
            b = 0.0
            a = client.rho_total / sum(rho_useds[i])
        else:
            ubs = [client.rho_total / CR_Total]
            lbs = [-np.inf]
            for CR in range(CR_Total):
                denom = rho_used_sums[i] - CR_Total * rho_useds[i][CR]
                if denom > 0.0000001:
                    ubs.append((rho_used_max * rho_used_sums[i] - client.rho_total * rho_useds[i][CR]) / denom)
                    lbs.append((rho_used_min * rho_used_sums[i] - client.rho_total * rho_useds[i][CR]) / denom)
                elif denom < -0.0000001:
                    ubs.append((rho_used_min * rho_used_sums[i] - client.rho_total * rho_useds[i][CR]) / denom)
                    lbs.append((rho_used_max * rho_used_sums[i] - client.rho_total * rho_useds[i][CR]) / denom)
            bub = min(ubs)
            blb = max(lbs)
            b   = -sum([(rho_useds[i][CR] * rho_used_sums[i] - client.rho_total) *
                        (CR_Total * rho_useds[i][CR] - rho_used_sums[i])
                        for CR in range(CR_Total)]) / \
                   sum([(CR_Total * rho_useds[i][CR] - rho_used_sums[i])**2
                        for CR in range(CR_Total)])
            b = max(blb, min(bub, b))
            a = (client.rho_total - b * CR_Total) / rho_used_sums[i]
        for CR in range(CR_Total):
            rho_useds[i][CR] = a * rho_useds[i][CR] + b
            sigmas[i][CR]    = rho2sigma(rho=rho_useds[i][CR], el=client.el, bs=client.bs)
    return sigmas, rho_useds


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    t0 = time.time()

    agent = Agent(n_agents=N, dim_state=2, dim_action=1)

    # Load weights ─────────────────────────────────────────────────────────────
    if LOAD_EPISODE is not None:
        print(f'[Load] Loading best model at episode {LOAD_EPISODE}.')
        agent.load(episode=LOAD_EPISODE)
    elif os.path.exists(CHECKPOINT_PATH):
        print(f'[Load] Loading checkpoint from {CHECKPOINT_PATH}.')
        agent.load_checkpoint(CHECKPOINT_PATH)
    else:
        raise FileNotFoundError(
            f'No model weights found. Either set LOAD_EPISODE or ensure '
            f'{CHECKPOINT_PATH} exists.')

    mp.set_start_method('spawn', force=True)

    os.makedirs('./exp_data', exist_ok=True)

    # ── Datasets ──────────────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))   # match improved train
    ])

    trainsets = []
    for i in range(torch.cuda.device_count()):
        trainset = CUDACIFAR10(root='./data', train=True, download=True,
                               pre_transform=transform, device='cuda:' + str(i))
        print(f'Training set size on cuda:{i} =', len(trainset))
        trainsets.append(trainset)

    testset = CUDACIFAR10_Test(root='./data', train=False, download=True,
                               pre_transform=transform, device='cuda:0')
    print('Test set size =', len(testset))

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=200, shuffle=True, num_workers=0, pin_memory=False)

    indices     = list(range(len(trainsets[0])))
    random.shuffle(indices)
    dataset_num = int(len(trainsets[0]) / N)
    trsets = []
    for i in range(N):
        device_index = i % torch.cuda.device_count()
        device       = 'cuda:' + str(device_index)
        trset        = torch.utils.data.Subset(
            trainsets[device_index], indices[i * dataset_num: (i + 1) * dataset_num])
        trsets.append([trset, device])

    # ── Server & Clients ──────────────────────────────────────────────────────
    net    = Net()
    server = Server(net=net)

    clients = [Client(id=i,
                      rho_total=np.random.uniform(low=rho_min, high=rho_max),
                      trset=trsets[i][0],
                      agent=agent,
                      device=trsets[i][1])
               for i in range(N)]

    # ── Accumulators across episodes ──────────────────────────────────────────
    accs1     = []   # DRL (ours)
    accs2     = []   # Uniform
    accs4     = []   # Exponential
    accs5     = []   # Arithmetic (Linear)
    rhos1, rhos2, rhos4, rhos5         = [], [], [], []
    sigmass1, sigmass2, sigmass4, sigmass5 = [], [], [], []

    episode = 0
    while episode < episodes:
        t1 = time.time()
        print('=' * 60)
        print(f'Episode {episode}')

        # Shared initialisation for this episode
        net_init = Net()
        net      = Net()
        net.load_state_dict(net_init.state_dict())
        server.reset(net=net)

        rhos_init = []
        for i, client in enumerate(clients):
            rho = np.random.uniform(low=rho_min, high=rho_max)
            rhos_init.append(rho)
            client.reset(rho_total=rho)

        # ── DRL policy test ───────────────────────────────────────────────────
        print('\nDRL test ================================================')
        net = Net()
        net.load_state_dict(net_init.state_dict())
        server.reset(net=net)
        for i, client in enumerate(clients):
            client.reset(rho_total=rhos_init[i])

        global_accs_DRL = [test(net=server.net, testloader=testloader, device=server.device)]

        sigmas    = [[] for _ in range(N)]
        rho_useds = [[] for _ in range(N)]
        rhos      = [[] for _ in range(N)]

        for CR in range(CR_Total):
            for i, client in enumerate(clients):
                state    = [CR / CR_Total, client.rho_total / rho_max]
                action   = agent.take_action(state=state)
                sigma    = action[0] * (sigma_max - sigma_min) + sigma_min
                rho_used = sigma2rho(sigma=sigma, el=client.el, bs=client.bs)
                sigmas[i].append(sigma)
                rho_useds[i].append(rho_used)

        sigmas, rho_useds = project_sigmas(sigmas, rho_useds, clients)

        # ── Sigma plot for three representative clients (low / mid / high rho) ─
        sorted_clients = sorted(range(N), key=lambda i: clients[i].rho_total)
        idx0, idx1, idx2 = sorted_clients[0], sorted_clients[N // 2], sorted_clients[-1]
        client0, client1, client2 = clients[idx0], clients[idx1], clients[idx2]
        sigmas0, sigmas1_plot, sigmas2 = sigmas[idx0], sigmas[idx1], sigmas[idx2]

        plt.clf()
        plt.plot(range(CR_Total), sigmas0,       linestyle='solid')
        plt.plot(range(CR_Total), sigmas1_plot,  linestyle='dotted')
        plt.plot(range(CR_Total), sigmas2,       linestyle='dashed')
        plt.xticks([0, 5, 10, 15, 20, 25], size=12)
        plt.yticks(size=12)
        plt.xlabel('Communication Round $t$',    fontsize=16)
        plt.ylabel('Noise Level $\\sigma_n^t$',  fontsize=16)
        plt.legend([
            'Client with ' + r'$\rho_k=$' + f'{client0.rho_total:.5f}',
            'Client with ' + r'$\rho_k=$' + f'{client1.rho_total:.5f}',
            'Client with ' + r'$\rho_k=$' + f'{client2.rho_total:.5f}',
        ], fontsize=14)
        plt.grid()
        plt.tight_layout()
        plt.savefig('./exp_data/sigmas_DRL_test_CIFAR10_paired.pdf')
        print('Sigma schedule saved.')

        # Run federated rounds with DRL sigmas
        for i, client in enumerate(clients):
            client.reset(rho_total=rhos_init[i])

        for CR in range(CR_Total):
            for i, client in enumerate(clients):
                rhos[i].append(client.rho)
                client.sigma = sigmas[i][CR]
                client.train(net=server.net, sigma=client.sigma)
            net_list = []
            for client in clients:
                net_list.append(client.outs.get())
            server.aggregate(net_list=net_list)
            global_accs_DRL.append(test(net=server.net, testloader=testloader, device=server.device))

        for i, client in enumerate(clients):
            rhos[i].append(client.rho)

        accs1.append(global_accs_DRL)
        rhos1.append(rhos)
        sigmass1.append(sigmas)

        # ── Uniform baseline ──────────────────────────────────────────────────
        print('\nUniform baseline ========================================')
        sigma_Uniform = []
        sigmas    = [[] for _ in range(N)]
        rhos      = [[] for _ in range(N)]
        for i, client in enumerate(clients):
            sigma = rho2sigma(rho=client.rho_total / CR_Total, el=client.el, bs=client.bs)
            sigma_Uniform.append(sigma)
            sigmas[i] = [sigma] * CR_Total

        net = Net()
        net.load_state_dict(net_init.state_dict())
        server.reset(net=net)
        for i, client in enumerate(clients):
            client.reset(rho_total=rhos_init[i])

        global_accs_Uniform = [test(net=server.net, testloader=testloader, device=server.device)]
        for CR in range(CR_Total):
            for i, client in enumerate(clients):
                rhos[i].append(client.rho)
                client.train(net=server.net, sigma=sigma_Uniform[i])
            net_list = [client.outs.get() for client in clients]
            server.aggregate(net_list=net_list)
            global_accs_Uniform.append(test(net=server.net, testloader=testloader, device=server.device))
        for i, client in enumerate(clients):
            rhos[i].append(client.rho)

        accs2.append(global_accs_Uniform)
        rhos2.append(rhos)
        sigmass2.append(sigmas)

        # ── Exponential baseline ──────────────────────────────────────────────
        print('\nExponential baseline ====================================')
        net = Net()
        net.load_state_dict(net_init.state_dict())
        server.reset(net=net)
        for i, client in enumerate(clients):
            client.reset(rho_total=rhos_init[i])

        sigmas    = [[] for _ in range(N)]
        rho_useds = [[] for _ in range(N)]
        rhos      = [[] for _ in range(N)]
        decay_ratio = 2**(1 / (CR_Total - 1))

        for i, client in enumerate(clients):
            rho_useds[i].append((1 - decay_ratio) * client.rho_total / (1 - decay_ratio**CR_Total))
            for CR in range(CR_Total):
                rho_used = rho_useds[i][0] * (decay_ratio**CR)
                sigmas[i].append(rho2sigma(rho=rho_used, el=client.el, bs=client.bs))

        global_accs_Exp = [test(net=server.net, testloader=testloader, device=server.device)]
        for CR in range(CR_Total):
            for i, client in enumerate(clients):
                rhos[i].append(client.rho)
                client.train(net=server.net, sigma=sigmas[i][CR])
            net_list = [client.outs.get() for client in clients]
            server.aggregate(net_list=net_list)
            global_accs_Exp.append(test(net=server.net, testloader=testloader, device=server.device))
        for i, client in enumerate(clients):
            rhos[i].append(client.rho)

        accs4.append(global_accs_Exp)
        rhos4.append(rhos)
        sigmass4.append(sigmas)

        # ── Arithmetic (Linear) baseline ──────────────────────────────────────
        print('\nArithmetic baseline =====================================')
        net = Net()
        net.load_state_dict(net_init.state_dict())
        server.reset(net=net)
        for i, client in enumerate(clients):
            client.reset(rho_total=rhos_init[i])

        sigmas    = [[] for _ in range(N)]
        rho_useds = [[] for _ in range(N)]
        rhos      = [[] for _ in range(N)]
        decays    = [2 * client.rho_total / (CR_Total * (CR_Total - 1) * 3) for client in clients]

        for i, client in enumerate(clients):
            rho_useds[i].append(2 / 3 * (client.rho_total / CR_Total))
            for CR in range(CR_Total):
                rho_used = rho_useds[i][0] + decays[i] * CR
                sigmas[i].append(rho2sigma(rho=rho_used, el=client.el, bs=client.bs))

        global_accs_Linear = [test(net=server.net, testloader=testloader, device=server.device)]
        for CR in range(CR_Total):
            for i, client in enumerate(clients):
                rhos[i].append(client.rho)
                client.train(net=server.net, sigma=sigmas[i][CR])
            net_list = [client.outs.get() for client in clients]
            server.aggregate(net_list=net_list)
            global_accs_Linear.append(test(net=server.net, testloader=testloader, device=server.device))
        for i, client in enumerate(clients):
            rhos[i].append(client.rho)

        accs5.append(global_accs_Linear)
        rhos5.append(rhos)
        sigmass5.append(sigmas)

        print(f'\nTime for episode {episode}: {time.time() - t1:.1f}s')
        episode += 1

    # ── Aggregate results across episodes ─────────────────────────────────────
    avg_accs1 = [np.mean([l[CR] for l in accs1]) for CR in range(CR_Total + 1)]
    avg_accs2 = [np.mean([l[CR] for l in accs2]) for CR in range(CR_Total + 1)]
    avg_accs4 = [np.mean([l[CR] for l in accs4]) for CR in range(CR_Total + 1)]
    avg_accs5 = [np.mean([l[CR] for l in accs5]) for CR in range(CR_Total + 1)]

    print('\n' + '=' * 60)
    print('Final accuracies (averaged across episodes):')
    print(f'  DRL (ours)  : {avg_accs1[-1]:.4f}')
    print(f'  Uniform     : {avg_accs2[-1]:.4f}')
    print(f'  Exponential : {avg_accs4[-1]:.4f}')
    print(f'  Arithmetic  : {avg_accs5[-1]:.4f}')

    # ── PAIRED STATISTICS (the point of this harness) ─────────────────────────
    # Every arm in a trial shares one init (net_init) and one set of client
    # budgets (rhos_init), so differences are paired and init luck cancels.
    from scipy import stats as _st
    _fin = {'DRL': np.array([l[-1] for l in accs1], float),
            'Uniform': np.array([l[-1] for l in accs2], float),
            'Exponential': np.array([l[-1] for l in accs4], float),
            'Arithmetic': np.array([l[-1] for l in accs5], float)}
    _n = len(_fin['DRL'])
    print('\n' + '=' * 60)
    print(f'PAIRED COMPARISON, n = {_n} trials (final-round accuracy)')
    print(f'{"arm":<13}{"mean":>9}{"sd":>9}')
    for _k, _v in _fin.items():
        print(f'{_k:<13}{_v.mean():>9.4f}{(_v.std(ddof=1) if _n > 1 else 0):>9.4f}')
    print(f'\n{"DRL vs":<13}{"mean diff":>11}{"95% CI":>22}{"wins":>8}{"p":>9}')
    _out = {'n': _n, 'per_trial': {k: v.tolist() for k, v in _fin.items()}, 'tests': {}}
    for _k in ['Uniform', 'Exponential', 'Arithmetic']:
        _d = _fin['DRL'] - _fin[_k]
        if _n > 1:
            _ci = _st.t.interval(0.95, _n - 1, loc=_d.mean(), scale=_st.sem(_d))
            _p = float(_st.ttest_rel(_fin['DRL'], _fin[_k])[1])
        else:
            _ci, _p = (float('nan'), float('nan')), float('nan')
        print(f'{_k:<13}{_d.mean():>+11.4f}   [{_ci[0]:+.4f},{_ci[1]:+.4f}]{int((_d > 0).sum()):>6}/{_n}{_p:>9.3f}')
        _out['tests'][_k] = {'mean_diff': float(_d.mean()), 'ci95': [float(_ci[0]), float(_ci[1])],
                             'wins': int((_d > 0).sum()), 'p': _p,
                             'sd_paired_diff': float(_d.std(ddof=1)) if _n > 1 else None}
    print('\nInterpretation: a CI that contains 0 means this experiment cannot')
    print('distinguish the arms. Compare the sd of the paired difference against')
    print('the effect the source paper reports (CIFAR-10: +0.0015).')
    json.dump(_out, open('./exp_data/CIFAR10_paired_stats.json', 'w'), indent=1)
    print('\nwrote ./exp_data/CIFAR10_paired_stats.json')

    # ── Accuracy curve plot ───────────────────────────────────────────────────
    plt.clf()
    plt.plot(range(CR_Total + 1), avg_accs1, linestyle='solid')
    plt.plot(range(CR_Total + 1), avg_accs2, linestyle='dotted')
    plt.plot(range(CR_Total + 1), avg_accs4, linestyle='dashed')
    plt.plot(range(CR_Total + 1), avg_accs5, linestyle='dashdot')
    plt.xticks([0, 5, 10, 15, 20, 25])
    plt.xlabel('Communication Round $t$', fontsize=12)
    plt.ylabel('Test Accuracy',           fontsize=12)
    plt.legend(['DRL (ours)', 'Uniform', 'Exponential', 'Arithmetic'])
    plt.grid()
    plt.tight_layout()
    plt.savefig('./exp_data/Acc_Test_CIFAR10_paired.pdf')
    print('Accuracy curve saved to ./exp_data/Acc_Test_CIFAR10_improved.pdf')

    # ── Save experimental data ────────────────────────────────────────────────
    exp_data = [accs1, accs2, accs4, accs5,
                rhos1, rhos2, rhos4, rhos5,
                sigmass1, sigmass2, sigmass4, sigmass5]
    exp_data_json = json.dumps(exp_data)
    with open('./exp_data/exp_data_CIFAR10_paired.json', 'w') as fp:
        json.dump(exp_data_json, fp)
    print('Experimental data saved to ./exp_data/exp_data_CIFAR10_improved.json')

    # ── Teardown ──────────────────────────────────────────────────────────────
    for client in clients:
        client.terminate_process()
    for client in clients:
        client.process.join()

    print(f'\nTotal time: {time.time() - t0:.1f}s')
