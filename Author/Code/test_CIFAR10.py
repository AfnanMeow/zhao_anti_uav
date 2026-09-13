from collections.abc import Callable, Iterable, Mapping
from typing import Any
import numpy as np
import torch.multiprocessing as mp
import time
import random
import torch
from typing import Any, Callable, Optional, Tuple
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import opacus
import numpy as np
from collections import deque
import random
import os
import copy
import warnings
import pickle
import matplotlib.pyplot as plt
import matplotlib
import json
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

N = 10 # Number of clients
batch_size = 200
episodes = 1 
CR_Total = 25
his_len = 3
rho_min = 20.0
rho_max = 40.0
epoch_local = 4
DRL_steps = 1
loss_max = -np.log(0.1)
sigma_max = 20.0
sigma_min = 1.0
rho_used_min = 2 / sigma_max**2 * epoch_local
rho_used_max = 2 / sigma_min**2 * epoch_local

def rho2sigma(rho, el, bs):
    return np.sqrt((2 * el)/(bs * rho))

def sigma2rho(sigma, el, bs):
    return 2 * el / (bs * (sigma**2))

class CUDACIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, 
                 root: str, 
                 train: bool = True, 
                 pre_transform: Callable[..., Any]  = None,
                 transform: Callable[..., Any] = None, 
                 target_transform: Callable[..., Any] = None, 
                 download: bool = False,
                 device: str = 'cuda:0') -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.data = self.data.astype('float32')
        for i in range(len(self)):
            self.data[i] = pre_transform(self.data[i]/255.0).numpy().transpose((1, 2, 0))
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
    def __init__(self, 
                 root: str, 
                 train: bool = False, 
                 pre_transform: Callable[..., Any]  = None,
                 transform: Callable[..., Any] = None, 
                 target_transform: Callable[..., Any] = None, 
                 download: bool = False,
                 device: str = 'cuda:0') -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.data = self.data.astype('float32')
        for i in range(len(self)):
            self.data[i] = pre_transform(self.data[i]/255.0).numpy().transpose((1, 2, 0))
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

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
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
    

class Agent():
    def __init__(self,
                 n_agents,
                 dim_state,
                 dim_action,
                 lr_actor = 0.000015,
                 lr_critic = 0.00015,
                 lr_actor_target=0.005,
                 lr_critic_target=0.005,
                 buffer_size=4000,
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
        
        # Parameter sharing actor
        self.actor = Actor(dim_state=self.dim_state, dim_action=self.dim_action).cuda(device=device)
        self.actor_target = Actor(dim_state=self.dim_state, dim_action=self.dim_action).cuda(device=device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Parameter sharing critic
        self.critic = Critic(dim_state=self.dim_state, dim_action=self.dim_action).cuda(device=device)
        self.critic_target = Critic(dim_state=self.dim_state, dim_action=self.dim_action).cuda(device=device)
        for i in range(self.n_agents):
            self.critic_target.load_state_dict(self.critic.state_dict())
            
        # Network optimizer hyperparameters
        self.lr_actor_target = lr_actor_target
        self.lr_critic_target = lr_critic_target
        self.bs = bs
        self.lr_actor_target = lr_actor_target
        self.criterion_critic = nn.MSELoss()
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Replay buffer
        self.minimal_size = minimal_size
        self.rep_buffer = {'state' : [deque(maxlen=buffer_size) for _ in range(self.n_agents)],
                       'action' : [deque(maxlen=buffer_size) for _ in range(self.n_agents)],
                       'reward' : deque(maxlen=buffer_size),
                       'next_state' : [deque(maxlen=buffer_size) for _ in range(self.n_agents)],
                       'done' : deque(maxlen=buffer_size)}
        
        # Discount factor
        self.gamma = torch.Tensor([gamma]).cuda(device=device)
        
        # Initial explore rate
        self.explore = explore
        
        # Explore decay
        self.explore_decay = explore_decay
        self.explore_decay_interval = explore_decay_interval
        self.step = 1
        self.train_step = 1
        self.train_interval = train_interval

    def train(self, state, action, reward, next_state, done):
        for i in range(self.n_agents):
            self.rep_buffer['state'][i].append(state[i])
            self.rep_buffer['action'][i].append([action[i]])
            self.rep_buffer['next_state'][i].append(next_state[i])
        self.rep_buffer['reward'].append(reward)
        self.rep_buffer['done'].append(done)
        if (len(self.rep_buffer['state'][0]) >= self.minimal_size):
            self.explore = max(self.explore * self.explore_decay, 0.01)
            if self.train_step % self.train_interval == 0:
                for step in range(DRL_steps):
                    indices = list(range(len(self.rep_buffer['state'][0])))
                    random.shuffle(indices)
                    state = [[self.rep_buffer['state'][i][index] for index in indices[:self.bs]] for i in range(self.n_agents)]
                    action = [[self.rep_buffer['action'][i][index] for index in indices[:self.bs]] for i in range(self.n_agents)]
                    reward = [self.rep_buffer['reward'][index] for index in indices[:self.bs]]
                    next_state = [[self.rep_buffer['next_state'][i][index] for index in indices[:self.bs]] for i in range(self.n_agents)]
                    done = [self.rep_buffer['done'][index] for index in indices[:self.bs]]
                    state = torch.Tensor(state).cuda(device=self.device)
                    action = torch.Tensor(action).cuda(device=self.device)
                    reward = torch.Tensor(reward).cuda(device=self.device)
                    next_state = torch.Tensor(next_state).cuda(device=self.device)
                    done = torch.Tensor(done).cuda(device=self.device)
                    with torch.no_grad():
                        next_action = [self.actor_target(next_state[i]) for i in range(self.n_agents)]
                        Q_next = [self.critic_target(torch.cat([next_state[i], next_action[i]], dim=1)) for i in range(self.n_agents)]
                        # Q_tot_target =  reward_acc * self.n_agents + sum([self.gamma * Q_next[i] * (1.0 - done) for i in range(self.n_agents)])
                        Q_target =  reward + sum([self.gamma * Q_next[i] * (1.0 - done) for i in range(self.n_agents)])


                    # Update Q-functions
                    self.optimizer_critic.zero_grad()
                    Q_losses = []
                    for i in range(self.n_agents):
                        Q_pred = self.critic(torch.cat([state[i], action[i]], dim=1))
                        Q_losses.append(Q_pred)
                    Q_loss = self.criterion_critic(sum(Q_losses), Q_target)
                    Q_loss.backward()
                    self.optimizer_critic.step()

                    # Update actor
                    self.optimizer_actor.zero_grad()
                    self.optimizer_critic.zero_grad()
                    A_losses = []
                    for i in range(self.n_agents):
                        A_loss = self.critic(torch.cat([state[i], self.actor(state[i])], dim=1))
                        A_losses.append(A_loss)
                    
                    A_loss = -torch.mean(sum(A_losses) / self.n_agents)
                    # A_loss.backward(retain_graph=True)
                    A_loss.backward()
                    self.optimizer_actor.step()
                    self.step += 1
                
                

                # Update target networks
                for name, p in self.actor_target.named_parameters():
                    p.data = (1 - self.lr_actor_target) * p.data + self.lr_actor_target * self.actor.state_dict()[name].data.clone()
                for name, p in self.critic_target.named_parameters():
                    p.data = (1 - self.lr_critic_target) * p.data + self.lr_critic_target * self.critic.state_dict()[name].data.clone()
                    
                
            self.train_step += 1

    def take_action(self, state):
        with torch.no_grad():
            action = self.actor(torch.Tensor(state).cuda(device=self.device)).cpu().numpy()
            return action
        
    def take_action_with_noise(self, state):
        with torch.no_grad():
            action = self.actor(torch.Tensor(state).cuda(device=self.device)).cpu().numpy()
            explore = self.explore
            action_noise = action + np.random.normal(loc=0.0, scale=explore, size=self.dim_action)
            for i in range(len(action_noise)):
                local_explore = explore
                while action_noise[i] < 0.0 or action_noise[i] > 1.0:
                    local_explore /= 2
                    if local_explore <= 0.0001:
                        action_noise[i] = action[i]
                        break
                    action_noise[i] = np.random.normal(loc=action_noise[i], scale=local_explore, size=1)
            return action, action_noise

    def save(self, episode):
        torch.save(self.actor.state_dict(), './DRL_Nets/actor_CIFAR10_'+ str(episode) + '.pth')
        torch.save(self.actor_target.state_dict(), './DRL_Nets/actor_target_CIFAR10_'+ str(episode) + '.pth')
        torch.save(self.critic.state_dict(), './DRL_Nets/critic_CIFAR10_'+ str(episode) + '.pth')
        torch.save(self.critic_target.state_dict(), './DRL_Nets/critic_target_CIFAR10_'+ str(episode) + '.pth')
        
    def load(self, episode):
        self.actor.load_state_dict(torch.load('./DRL_Nets/actor_CIFAR10_'+ str(episode) + '.pth', map_location=self.device))
        self.actor_target.load_state_dict(torch.load('./DRL_Nets/actor_target_CIFAR10_'+ str(episode) + '.pth', map_location=self.device))
        self.critic.load_state_dict(torch.load('./DRL_Nets/critic_CIFAR10_'+ str(episode) + '.pth', map_location=self.device))
        self.critic_target.load_state_dict(torch.load('./DRL_Nets/critic_target_CIFAR10_'+ str(episode) + '.pth', map_location=self.device))

class Training(mp.Process):
    def __init__(self, C=4, bs=batch_size, lr=0.1, el=epoch_local, ins=mp.Queue(),
                 outs=mp.Queue(), device='cuda:0') -> None:
        mp.Process.__init__(self)
        
        # Client parameters
        self.device = device
        self.C = C
        self.bs = bs
        self.el = el
        self.lr = lr
        
        # Processes communication
        self.ins = ins
        self.outs = outs
        
        

    def run(self):
        # Initiate local state
        trset = self.ins.get()
        trainloader = torch.utils.data.DataLoader(trset, batch_size=self.bs,
                                          shuffle=True, num_workers=0, pin_memory=False)
        self.net = Net()
        pe = opacus.PrivacyEngine()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr)
        self.net, self.optimizer, _ = pe.make_private(
            module=self.net,
            data_loader=trainloader,
            optimizer=self.optimizer,
            noise_multiplier=1.0,
            max_grad_norm=self.C
        )
        self.net.cuda(device=self.device)
        print('Dataset loaded successfully of', self.device)
        self.outs.put(1)
        
        while True:
            mode, net, sigma = self.ins.get()
            if mode == 0: # Terminate mode
                break
            elif mode == 1: # training mode
                self.net.load_state_dict(net.state_dict())
                self.optimizer.noise_multiplier = sigma
                for epoch in range(self.el):
                    for i, data in enumerate(trainloader):
                        self.optimizer.zero_grad()
                        inputs, labels = data
                        outputs = self.net(inputs)
                        loss = self.criterion(outputs, labels)
                        loss.backward()
                        self.optimizer.step()
                self.outs.put(self.net)
            else: # Testing mode
                self.net.load_state_dict(net.state_dict())
                with torch.no_grad():
                    correct = 0
                    total = 0
                    avg_loss = 0
                    count = 0
                    for i, data in enumerate(trainloader):
                        inputs, labels = data
                        outputs = self.net(inputs)
                        loss = self.criterion(outputs, labels)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        avg_loss = avg_loss * (count / (count + 1)) + loss.item() / (count + 1)
                        count += 1
                    acc = correct / total
                    self.outs.put([acc, avg_loss])

class Client():
    def __init__(self, id=-1, C=6, rho_total=20.0,
                 bs=batch_size, trset=None, lr=0.1, el=epoch_local, hl=his_len, agent=None, device='cuda:0') -> None:
        # Basic client status
        self.id = id
        self.rho = rho_total
        self.rho_total = rho_total
        self.C = C
        self.bs = bs
        self.trset = trset
        self.lr = lr
        self.el = el
        self.hl = hl
        self.sigma = sigma_max
        self.rho_used = 2 * self.el / (self.sigma**2)
        self.last_local_acc = 0.1
        self.local_acc = 0.1
        self.isAvailable = True
        
        # DRL variables
        self.state = None
        self.action = None
        self.reward = None
        self.next_state = None
        self.done = None
        self.real_action = None
        self.acc = 0
        self.device = device
            
        self.agent = agent
        
        
            
        # Training process
        self.ins = mp.Queue(1)
        self.outs = mp.Queue(1)
        self.process = Training(C=self.C, bs=self.bs, lr=self.lr, el=self.el,
                     ins=self.ins, outs=self.outs, device=device)
        self.process.start()
        self.ins.put(self.trset)
        self.outs.get() # Waiting for data loading
        

    def reset(self, rho_total):
        self.rho = rho_total
        self.rho_total = rho_total
        self.action = None
        self.sigma = sigma_max
        self.isAvailable = True
        self.last_local_acc = 0.1
        self.local_acc = 0.1

        
    def train(self, net, sigma):
        self.sigma = sigma
        rho_per_epoch = 2 / (sigma**2)
        self.rho_used = self.el * rho_per_epoch
        self.rho -= self.rho_used
        self.ins.put([1, net, sigma])
        
    def test(self, net):
        self.ins.put([2, net, 0])
    
    def terminate_process(self):
        self.ins.put([0, 0, 0])
        

class Server():
    def __init__(self, net, device='cuda:0') -> None:
        self.device = device
        self.net = Net()
        self.net.load_state_dict(net.state_dict())
        dummyOptimizer = optim.SGD(self.net.parameters(), lr=0)
        dummySet = [[0, 0]]
        dummyLoader = torch.utils.data.DataLoader(dummySet)
        privacyEngine = opacus.PrivacyEngine()
        self.net, _, _ = privacyEngine.make_private(
            module=self.net,
            optimizer=dummyOptimizer,
            data_loader=dummyLoader,
            noise_multiplier=4,
            max_grad_norm=4
        )
        self.dummy_net = copy.deepcopy(self.net)
        self.dummy_net.cuda(device=self.device)
        self.net.cuda(device=self.device)

    def reset(self, net):
        self.net = Net()
        self.net.load_state_dict(net.state_dict())
        dummyOptimizer = optim.SGD(self.net.parameters(), lr=0)
        dummySet = [[0, 0]]
        dummyLoader = torch.utils.data.DataLoader(dummySet)
        privacyEngine = opacus.PrivacyEngine()
        self.net, _, _ = privacyEngine.make_private(
            module=self.net,
            optimizer=dummyOptimizer,
            data_loader=dummyLoader,
            noise_multiplier=4,
            max_grad_norm=4
        )
        self.net.cuda(device=self.device)

    def aggregate(self, net_list):
        count = 0
        for net in net_list:
            self.dummy_net.load_state_dict(net.state_dict())
            for name, p in self.net.named_parameters():
                p.data = p.data * (count / (count + 1)) + self.dummy_net.state_dict()[name].data / (count + 1)
            count += 1
            
def test(net, testloader, device):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.cuda(device=device)
            labels = labels.cuda(device=device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total
            
if __name__ == '__main__':
    t0 = time.time()
    agent = Agent(n_agents=N,
                  dim_state=2,
                  dim_action=1)
    agent.load(episode=350)
    mp.set_start_method('spawn', force=True)
    
    # Initialize server
    net = Net()
    server = Server(net=net)
    
    # Split Datasets
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))])
    batch_size = 200
    trainsets = []
    for i in range(torch.cuda.device_count()):
        trainset = CUDACIFAR10(root='./data', train=True,
                                        download=True, pre_transform=transform, device='cuda:' + str(i))
        print(len(trainset))
        trainsets.append(trainset)
    
    indices = list(range(len(trainsets[0])))
    random.shuffle(indices)
    dataset_num = int(len(trainsets[0]) / N)
    trsets = []
    
    testset = CUDACIFAR10_Test(root='./data', train=False,
                                        download=True, pre_transform=transform, device='cuda:0')
    print(len(testset))
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=200,
                                          shuffle=True, num_workers=0, pin_memory=False)
    
    for i in range(N):
        device_index = i % torch.cuda.device_count()
        device = 'cuda:' + str(device_index)
        trset = torch.utils.data.Subset(trainsets[device_index], indices[i * dataset_num: (i + 1) * dataset_num])
        trsets.append([trset, device])
    
    
    
    # Initialize clients
    clients = [Client(id=i, rho_total=np.random.uniform(low=rho_min, high=rho_max),
                    trset=trsets[i][0], agent=agent, device=trsets[i][1]) for i in range(N)]
    
    # Start training ##################################################################################################################
    episode = 0
    accs1 = []
    accs2 = []
    accs3 = []
    accs4 = []
    accs5 = []
    rhos1 = []
    rhos2 = []
    rhos3 = []
    rhos4 = []
    rhos5 = []
    sigmass1 = []
    sigmass2 = []
    sigmass3 = []
    sigmass4 = []
    sigmass5 = []
    while episode < episodes:
        t1 = time.time()
        print('Episode', episode)
        # Reset clients and server
        
        net_init = Net()
        net = Net()
        net.load_state_dict(net_init.state_dict())
        server.reset(net=net)
        rhos_init = []
        for i, client in enumerate(clients):
            rho = np.random.uniform(low=rho_min, high=rho_max)
            rhos_init.append(rho)
            client.reset(rho_total=rho)
            
        # Tests
        if episode % 1 == 0:
            # DRL test
            print('DRL test==========================================')
            net_init = Net()
            net = Net()
            net.load_state_dict(net_init.state_dict())
            server.reset(net=net)
                
            for i, client in enumerate(clients):
                client.reset(rho_total=rhos_init[i])
            global_accs_DRL = []
            global_accs_DRL.append(test(net=server.net, testloader=testloader, device=server.device))
            
            sigmas = [[] for _ in range(N)]
            rho_useds = [[] for _ in range(N)]
            rhos = [[] for _ in range(N)]
            
            isDone = 0.0
            for CR in range(CR_Total):
                if CR == CR_Total - 1:
                    isDone = 1.0
                for i, client in enumerate(clients):
                    state = [CR / CR_Total,
                            client.rho_total / rho_max]
                    action = agent.take_action(state=state)
                    sigma = action[0] * (sigma_max - sigma_min) + sigma_min
                    rho_used = sigma2rho(sigma=sigma, el=client.el, bs=client.bs)
                    sigmas[i].append(sigma)
                    rho_useds[i].append(rho_used)
                
            rho_used_sums = [sum(rho_useds[i]) for i in range(N)]
            
            a = 1.0
            b = 0.0
            for i, client in enumerate(clients):
                if abs(max(sigmas[i]) - min(sigmas[i])) < 0.01:
                    b = 0.0
                    a = client.rho_total / sum(rho_useds[i])
                else:
                    ubs = []
                    lbs = []
                    ubs.append(client.rho_total / CR_Total)
                    lbs.append(-np.inf)
                    for CR in range(CR_Total):
                        if rho_used_sums[i] - CR_Total * rho_useds[i][CR] > 0.0000001:
                            ubs.append((rho_used_max * rho_used_sums[i] - client.rho_total * rho_useds[i][CR]) / (rho_used_sums[i] - CR_Total * rho_useds[i][CR]))
                            lbs.append((rho_used_min * rho_used_sums[i] - client.rho_total * rho_useds[i][CR]) / (rho_used_sums[i] - CR_Total * rho_useds[i][CR]))
                        elif rho_used_sums[i] - CR_Total * rho_useds[i][CR] < -0.0000001:
                            ubs.append((rho_used_min * rho_used_sums[i] - client.rho_total * rho_useds[i][CR]) / (rho_used_sums[i] - CR_Total * rho_useds[i][CR]))
                            lbs.append((rho_used_max * rho_used_sums[i] - client.rho_total * rho_useds[i][CR]) / (rho_used_sums[i] - CR_Total * rho_useds[i][CR]))
                    bub = min(ubs)
                    blb = max(lbs)
                    b = - sum([(rho_useds[i][CR]*rho_used_sums[i] - client.rho_total) * (CR_Total*rho_useds[i][CR] - rho_used_sums[i]) for CR in range(CR_Total)]) / \
                        (sum([(CR_Total*rho_useds[i][CR] - rho_used_sums[i])**2 for CR in range(CR_Total)]))
                    
                    if b > bub:
                        b = bub
                    elif b < blb:
                        b = blb
                    a = (client.rho_total - b * CR_Total) / rho_used_sums[i]
                for CR in range(CR_Total):
                    rho_useds[i][CR] = a * rho_useds[i][CR] + b
                    sigmas[i][CR] = rho2sigma(rho=rho_useds[i][CR], el=client.el, bs=client.bs)
            
            
            result_clients = [clients[0]]
            sigmas0 = []
            sigmas1 = []
            sigmas2 = []
            client0 = None
            client1 = None
            client2 = None
            if clients[0].rho_total <= clients[1].rho_total and clients[0].rho_total <= clients[2].rho_total:
                client0 = clients[0]
                sigmas0 = sigmas[0]
                if clients[1].rho_total <= clients[2].rho_total:
                    client1 = clients[1]
                    sigmas1 = sigmas[1]
                    client2 = clients[2]
                    sigmas2 = sigmas[2]
                else:
                    client1 = clients[2]
                    sigmas1 = sigmas[2]
                    client2 = clients[1]
                    sigmas2 = sigmas[1]
            elif clients[1].rho_total <= clients[0].rho_total and clients[1].rho_total <= clients[2].rho_total:
                client0 = clients[1]
                sigmas0 = sigmas[1]
                if clients[0].rho_total <= clients[2].rho_total:
                    client1 = clients[0]
                    sigmas1 = sigmas[0]
                    client2 = clients[2]
                    sigmas2 = sigmas[2]
                else:
                    client1 = clients[2]
                    sigmas1 = sigmas[2]
                    client2 = clients[0]
                    sigmas2 = sigmas[0]
            else:
                client0 = clients[2]
                sigmas0 = sigmas[2]
                if clients[0].rho_total <= clients[1].rho_total:
                    client1 = clients[0]
                    sigmas1 = sigmas[0]
                    client2 = clients[1]
                    sigmas2 = sigmas[1]
                else:
                    client1 = clients[1]
                    sigmas1 = sigmas[1]
                    client2 = clients[0]
                    sigmas2 = sigmas[0]
            plt.clf()
            plt.plot(list(range(CR_Total)), sigmas0, linestyle='solid')
            plt.plot(list(range(CR_Total)), sigmas1, linestyle='dotted')
            plt.plot(list(range(CR_Total)), sigmas2, linestyle='dashed')
            plt.xticks([0, 5, 10, 15, 20, 25], size=12)
            plt.yticks(size=12)
            plt.xlabel('Communication Round $t$', fontsize=16)
            plt.ylabel('Noise Level $\sigma_n^t$', fontsize=16)
            plt.legend(['Client with ' + r'$\rho_k=$' + "%.5f" % client0.rho_total,
                        'Client with ' + r'$\rho_k=$' + "%.5f" % client1.rho_total,
                        'Client with ' + r'$\rho_k=$' + "%.5f" % client2.rho_total], fontsize=14)
            plt.grid()
            plt.savefig('./exp_data/sigmas_DRL_test_CIFAR10.pdf')
            
            for i, client in enumerate(clients):
                client.reset(rho_total=rhos_init[i])
            
            for CR in range(CR_Total):
                for i, client in enumerate(clients):
                    rhos[i].append(client.rho)
                    sigma = sigmas[i][CR]
                    client.sigma = sigma
                    client.train(net=server.net, sigma=client.sigma)
                net_list = []
                for client in clients:
                    cnet = client.outs.get()
                    net_list.append(cnet)
                server.aggregate(net_list=net_list)
                global_accs_DRL.append(test(net=server.net, testloader=testloader, device=server.device))
            for i, client in enumerate(clients):
                rhos[i].append(client.rho)
            rhos1.append(rhos)
            accs1.append(global_accs_DRL)
            sigmass1.append(sigmas)
        # Comparative tests    
            # Uniform test ==============================================================================================
            sigma_Uniform = []
            sigmas = [[] for _ in range(N)]
            rhos = [[] for _ in range(N)]
            for i, client in enumerate(clients):
                sigma = rho2sigma(rho=client.rho_total / CR_Total, el=client.el, bs=client.bs)
                sigma_Uniform.append(sigma)
                sigmas[i] = [sigma for _ in range(CR_Total)]
            
            net = Net()
            net.load_state_dict(net_init.state_dict())
            server.reset(net=net)
                
            for i, client in enumerate(clients):
                client.reset(rho_total=rhos_init[i])
            global_accs_Uniform = []
            global_accs_Uniform.append(test(net=server.net, testloader=testloader, device=server.device))
            
            for CR in range(CR_Total):
                for i, client in enumerate(clients):
                    rhos[i].append(client.rho)
                    sigma = sigma_Uniform[i]
                    client.sigma = sigma
                    client.train(net=server.net, sigma=sigma)
                net_list = []
                for client in clients:
                    cnet = client.outs.get()
                    net_list.append(cnet)
                server.aggregate(net_list=net_list)
                global_accs_Uniform.append(test(net=server.net, testloader=testloader, device=server.device))
            for i, client in enumerate(clients):
                rhos[i].append(client.rho)
            rhos2.append(rhos)
            accs2.append(global_accs_Uniform)
            sigmass2.append(sigmas)
            
        # Exp test ==============================================================================================
            net = Net()
            net.load_state_dict(net_init.state_dict())
            server.reset(net=net)
            for i, client in enumerate(clients):
                client.reset(rho_total=rhos_init[i])
            
            sigmas = [[] for _ in range(N)]
            rho_useds = [[] for _ in range(N)]
            decay_ratio = 2**(1/(CR_Total-1))
            
            
            isDone = 0.0
            for i, client in enumerate(clients):
                rho_useds[i].append((1-decay_ratio)*client.rho_total/(1-decay_ratio**CR_Total))
                for CR in range(CR_Total):
                    rho_used = rho_useds[i][0] * (decay_ratio**CR)
                    sigma = rho2sigma(rho=rho_used, el=client.el, bs=client.bs)
                    sigmas[i].append(sigma)
            
            global_accs_Exp = []
            global_accs_Exp.append(test(net=server.net, testloader=testloader, device=server.device))
            
            for CR in range(CR_Total):
                for i, client in enumerate(clients):
                    rhos[i].append(client.rho)
                    sigma = sigmas[i][CR]
                    client.sigma = sigma
                    client.train(net=server.net, sigma=sigma)
                net_list = []
                for client in clients:
                    cnet = client.outs.get()
                    net_list.append(cnet)
                server.aggregate(net_list=net_list)
                global_accs_Exp.append(test(net=server.net, testloader=testloader, device=server.device))
            for i, client in enumerate(clients):
                rhos[i].append(client.rho)
            rhos4.append(rhos)
            accs4.append(global_accs_Exp)
            sigmass4.append(sigmas)
            
        # Linear test ==============================================================================================
            net = Net()
            net.load_state_dict(net_init.state_dict())
            server.reset(net=net)
            for i, client in enumerate(clients):
                client.reset(rho_total=rhos_init[i])
            
            sigmas = [[] for _ in range(N)]
            rho_useds = [[] for _ in range(N)]
            decays = [2 * client.rho_total / (CR_Total * (CR_Total - 1) * 3) for client in clients]
            
            isDone = 0.0
            for i, client in enumerate(clients):
                rho_useds[i].append(2 / 3 * (client.rho_total / CR_Total))
                for CR in range(CR_Total):
                    rho_used = rho_useds[i][0] + decays[i] * CR
                    sigma = rho2sigma(rho=rho_used, el=client.el, bs=client.bs)
                    sigmas[i].append(sigma)
                
            global_accs_Linear = []
            global_accs_Linear.append(test(net=server.net, testloader=testloader, device=server.device))
            
            for CR in range(CR_Total):
                for i, client in enumerate(clients):
                    rhos[i].append(client.rho)
                    sigma = sigmas[i][CR]
                    client.sigma = sigma
                    client.train(net=server.net, sigma=sigma)
                net_list = []
                for client in clients:
                    cnet = client.outs.get()
                    net_list.append(cnet)
                server.aggregate(net_list=net_list)
                global_accs_Linear.append(test(net=server.net, testloader=testloader, device=server.device))
            for i, client in enumerate(clients):
                rhos[i].append(client.rho)
            rhos5.append(rhos)
            accs5.append(global_accs_Linear)
            sigmass5.append(sigmas)
        avg_accs1 = [np.mean([l[CR] for l in accs1]) for CR in range(CR_Total+1)]
        avg_accs2 = [np.mean([l[CR] for l in accs2]) for CR in range(CR_Total+1)]
        # avg_accs3 = [np.mean([l[CR] for l in accs3]) for CR in range(CR_Total)]
        avg_accs4 = [np.mean([l[CR] for l in accs4]) for CR in range(CR_Total+1)]
        avg_accs5 = [np.mean([l[CR] for l in accs5]) for CR in range(CR_Total+1)]
        print('acc DRL:', avg_accs1[-1])
        print('acc Uniform:', avg_accs2[-1])
        # print('acc random:', avg_accs3[-1])
        print('acc Exp:', avg_accs4[-1])
        print('acc Arithmetic:', avg_accs5[-1])
        plt.clf()
        plt.plot(list(range(CR_Total+1)), avg_accs1, linestyle='solid')
        plt.plot(list(range(CR_Total+1)), avg_accs2, linestyle='dotted')
        # plt.plot(list(range(CR_Total)), avg_accs3)
        plt.plot(list(range(CR_Total+1)), avg_accs4, linestyle='dashed')
        plt.plot(list(range(CR_Total+1)), avg_accs5, linestyle='dashdot')
        
        plt.xticks([0, 5, 10, 15, 20, 25])
        plt.xlabel('Communication Round $t$', fontsize=12)
        plt.ylabel('Test Accuracy', fontsize=12)
        plt.legend(['Our method', 'Uniform', 'Exponential', 'Arithmetic sequence'])
        plt.savefig('./exp_data/Acc_Test_CIFAR10.pdf')
        print('Time spent for this episode', time.time() - t1)
        
        episode += 1
            
    # Save experimental data
    exp_data = [accs1, accs2, accs4, accs5,
                rhos1, rhos2, rhos4, rhos5,
                sigmass1, sigmass2, sigmass4, sigmass5]
    exp_data = json.dumps(exp_data)
    with open("./exp_data/exp_data_CIFAR10.json", "w") as fp:   #Pickling
        json.dump(exp_data, fp)
            
    # Terminate all clients
    for client in clients:
        client.terminate_process()
    for client in clients:
        client.process.join()

    print('Time spend', time.time() - t0)