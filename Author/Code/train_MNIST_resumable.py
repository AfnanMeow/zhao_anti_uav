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
import matplotlib.pyplot as plt

# ── CRASH-RESUME SUPPORT (added; training semantics unchanged) ───────────────
# Persists everything the loop needs to continue byte-identically: the four
# networks, both optimisers, the replay buffer, explore/train_step, the
# accuracy bookkeeping, and the RNG states of torch/cuda/numpy/random.
# Writes atomically (tmp + os.replace) so a power cut mid-save cannot corrupt
# an existing checkpoint.
RESUME_TAG   = os.environ.get('RESUME_TAG', 'MNIST')
RESUME_PATH  = os.environ.get('RESUME_PATH', './DRL_Nets/resume_%s.pt' % RESUME_TAG)
RESUME_EVERY = int(os.environ.get('CKPT_EVERY', 1))     # episodes between saves
RESUME_ON    = os.environ.get('RESUME', '1') != '0'
# Periodic actor snapshots, for evaluating a run across checkpoints
# (FINDINGS_checkpoint_dependence.md). They are written to SNAPSHOT_DIR, NOT
# to ./DRL_Nets/, because the test protocol picks the LATEST actor file in
# ./DRL_Nets/ and that must always be the authors' best-so-far save
# (mean_acc > high_acc). Snapshots sharing that folder and filename pattern
# would be picked instead.
SNAPSHOT_EVERY = int(os.environ.get('SNAPSHOT_EVERY', 50))   # 0 disables
SNAPSHOT_DIR   = os.environ.get('SNAPSHOT_DIR', './DRL_Nets/snapshots')


def snapshot_save(agent, ep):
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    for name, net in (('actor', agent.actor), ('actor_target', agent.actor_target),
                      ('critic', agent.critic), ('critic_target', agent.critic_target)):
        torch.save(net.state_dict(),
                   os.path.join(SNAPSHOT_DIR, '%s_%d.pth' % (name, ep)))


def _buf_to_lists(buf):
    return {k: ([list(d) for d in v] if isinstance(v, list) else list(v))
            for k, v in buf.items()}


def _buf_maxlen(buf):
    for v in buf.values():
        d = v[0] if isinstance(v, list) else v
        if hasattr(d, 'maxlen') and d.maxlen:
            return d.maxlen
    return 4000


def _buf_from_lists(buf, saved, maxlen):
    # Use the LIVE buffer's shape as ground truth. Do NOT infer it from the
    # saved data: 'reward' and 'done' store single-element lists, so an
    # element-type test misclassifies them as per-agent and rebuilds the wrong
    # nesting -- which then breaks the next save with
    # "'list' object has no attribute 'maxlen'".
    from collections import deque
    for k, v in saved.items():
        if isinstance(buf[k], list):          # per-agent: list of deques
            buf[k] = [deque(x, maxlen=maxlen) for x in v]
        else:                                  # flat deque
            buf[k] = deque(v, maxlen=maxlen)
    return buf


def resume_save(agent, episode, final_accs, mean_final_accs, returns,
                mean_returns, high_acc, path=None):
    path = path or RESUME_PATH
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    blob = {
        'episode': episode,
        'actor': agent.actor.state_dict(),
        'actor_target': agent.actor_target.state_dict(),
        'critic': agent.critic.state_dict(),
        'critic_target': agent.critic_target.state_dict(),
        'opt_actor': agent.optimizer_actor.state_dict(),
        'opt_critic': agent.optimizer_critic.state_dict(),
        'explore': agent.explore,
        'train_step': agent.train_step,
        'rep_buffer': _buf_to_lists(agent.rep_buffer),
        'buffer_size': _buf_maxlen(agent.rep_buffer),
        'final_accs': final_accs,
        'mean_final_accs': mean_final_accs,
        'returns': returns,
        'mean_returns': mean_returns,
        'high_acc': high_acc,
        'rng_torch': torch.get_rng_state(),
        'rng_cuda': (torch.cuda.get_rng_state_all()
                     if torch.cuda.is_available() else None),
        'rng_numpy': np.random.get_state(),
        'rng_random': random.getstate(),
    }
    tmp = path + '.tmp'
    torch.save(blob, tmp)
    os.replace(tmp, path)          # atomic on POSIX and on NTFS


def resume_load(agent, path=None):
    """Returns (episode, final_accs, mean_final_accs, returns, mean_returns,
    high_acc) or None when there is nothing to resume from."""
    path = path or RESUME_PATH
    if not (RESUME_ON and os.path.exists(path)):
        return None
    b = torch.load(path, map_location='cuda:0' if torch.cuda.is_available() else 'cpu',
                   weights_only=False)
    agent.actor.load_state_dict(b['actor'])
    agent.actor_target.load_state_dict(b['actor_target'])
    agent.critic.load_state_dict(b['critic'])
    agent.critic_target.load_state_dict(b['critic_target'])
    agent.optimizer_actor.load_state_dict(b['opt_actor'])
    agent.optimizer_critic.load_state_dict(b['opt_critic'])
    agent.explore = b['explore']
    agent.train_step = b['train_step']
    _buf_from_lists(agent.rep_buffer, b['rep_buffer'], b['buffer_size'])
    torch.set_rng_state(b['rng_torch'].cpu() if hasattr(b['rng_torch'], 'cpu')
                        else b['rng_torch'])
    if b['rng_cuda'] is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([s.cpu() if hasattr(s, 'cpu') else s
                                          for s in b['rng_cuda']])
        except Exception as e:
            print('  (cuda RNG not restored: %s)' % e, flush=True)
    np.random.set_state(b['rng_numpy'])
    random.setstate(b['rng_random'])
    print('=' * 62, flush=True)
    print('RESUMED from %s at episode %d  (explore=%.4f, train_step=%d, '
          'buffer=%d, high_acc=%.4f)'
          % (path, b['episode'], b['explore'], b['train_step'],
             len(b['rep_buffer']['reward']), b['high_acc']), flush=True)
    print('=' * 62, flush=True)
    return (b['episode'], b['final_accs'], b['mean_final_accs'],
            b['returns'], b['mean_returns'], b['high_acc'])
# ─────────────────────────────────────────────────────────────────────────────

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # The server has four graphics cards; choose the one that is currently available.

# Some hyperparameters
N = 10 # Number of clients participating in federated learning
batch_size = 200 # Batch size updated locally by the clients
episodes = 800 # Number of episodes in reinforcement learning
CR_Total = 20 # Communication rounds in federated learning within an episode
rho_min = 2.0 # Minimum total privacy budget initialized randomly by the user
rho_max = 4.0 # The maximum total privacy budget randomly initialized by the user
epoch_local = 4 # Number of epochs in a local update
DRL_steps = 1 # The number of times the network is updated after collecting training data once in reinforcement learning.
sigma_max = 20.0 # The maximum selected sigma
sigma_min = 1.0 # The minimum selected sigma
rho_used_min = 2 / sigma_max**2 * epoch_local # rho corresponding to the maximum sigma
rho_used_max = 2 / sigma_min**2 * epoch_local # rho corresponding to the minimum sigma

def rho2sigma(rho, el, bs):
    return np.sqrt((2 * el)/(bs * rho))

def sigma2rho(sigma, el, bs):
    return 2 * el / (bs * (sigma**2))

# A subclass of the MNIST dataset, pre-stores the dataset in GPU memory and performs preprocessing
class CUDAMNIST(torchvision.datasets.MNIST):
    def __init__(self, 
                 root: str, 
                 train: bool = True, 
                 pre_transform: Callable[..., Any]  = None,
                 transform: Callable[..., Any] = None, 
                 target_transform: Callable[..., Any] = None, 
                 download: bool = False,
                 device: str = 'cuda:0') -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.data = self.data.type(torch.FloatTensor)
        for i in range(len(self)):
            self.data[i] = pre_transform((self.data[i]/255.0).numpy())
            self.targets[i] = torch.Tensor([self.targets[i]]).squeeze_().long()
        self.data = self.data.unsqueeze_(1).cuda(device=device)
        self.targets = self.targets.cuda(device=device)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

# Federated learning neural network model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 4)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
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

# The Actor network, in which dim_state represents the state dimension, and dim_action represents the action dimension
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

# The Critic network, in which dim_state represents the state dimension, and dim_action represents the action dimension
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
    
# Reinforcement learning DDPG model
class Agent():
    def __init__(self,
                 n_agents, # Number of Federated Learning Clients
                 dim_state, # Dimension of the state
                 dim_action, # Dimension of the action
                 lr_actor = 0.00002, # actor learning rate
                 lr_critic = 0.0002, # critic’s learning rate
                 lr_actor_target=0.005, # actor target network update rate
                 lr_critic_target=0.005, # update rate of the critic target network
                 buffer_size=4000, # Buffer size of the DDPG
                 minimal_size=512, # When the number of trajectories in the buffer exceeds this threshold, the network begins to update.
                 gamma=0.998, # discount factor
                 explore=1.0, # Initial tendency to explore
                 explore_decay=0.9996, # The decay rate of exploratory tendencies
                 explore_decay_interval=1000, # The decay interval of the exploration tendency (unused)
                 bs=256, # Batch size for a single DDPG update
                 train_interval=1, # DDPG update interval, which affects update frequency
                 device='cuda:0' # Selected GPU device
                 ) -> None:
        
        self.device = device
        self.n_agents = n_agents
        self.dim_state = dim_state
        self.dim_action = dim_action
        
        # Initialization of parameter-sharing actors and the target network
        self.actor = Actor(dim_state=self.dim_state, dim_action=self.dim_action).cuda(device=device)
        self.actor_target = Actor(dim_state=self.dim_state, dim_action=self.dim_action).cuda(device=device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Parameter-sharing critic and target network initialization
        self.critic = Critic(dim_state=self.dim_state, dim_action=self.dim_action).cuda(device=device)
        self.critic_target = Critic(dim_state=self.dim_state, dim_action=self.dim_action).cuda(device=device)
        for i in range(self.n_agents):
            self.critic_target.load_state_dict(self.critic.state_dict())
            
        # Network optimization parameters
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
        
        # Initial explore rate, which influences the intensity of Gaussian noise during exploration
        self.explore = explore
        
        # Explore decay
        self.explore_decay = explore_decay
        self.explore_decay_interval = explore_decay_interval
        self.step = 1
        self.train_step = 1
        self.train_interval = train_interval

    # Add a trajectory to the replay buffer, 
    # and start updating the DDPG network if the buffer size is greater than or equal to minimal_size.
    def train(self, state, action, reward, next_state, done):
        # Add trajectory
        for i in range(self.n_agents):
            self.rep_buffer['state'][i].append(state[i])
            self.rep_buffer['action'][i].append([action[i]])
            self.rep_buffer['next_state'][i].append(next_state[i])
        self.rep_buffer['reward'].append(reward)
        self.rep_buffer['done'].append(done)
        
        # Training begins once the conditions are met
        if (len(self.rep_buffer['state'][0]) >= self.minimal_size):
            self.explore = max(self.explore * self.explore_decay, 0.01)
            if self.train_step % self.train_interval == 0:
                for step in range(DRL_steps):
                    # Randomly select a batch from the replay buffer
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
                    
                    # Calculate Q_target
                    with torch.no_grad():
                        next_action = [self.actor_target(next_state[i]) for i in range(self.n_agents)]
                        Q_next = [self.critic_target(torch.cat([next_state[i], next_action[i]], dim=1)) for i in range(self.n_agents)]
                        Q_target =  reward + sum([self.gamma * Q_next[i] * (1.0 - done) for i in range(self.n_agents)])


                    # Update Q-function
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
                    A_loss.backward()
                    self.optimizer_actor.step()
                    self.step += 1
                
                

                # Update target networks
                for name, p in self.actor_target.named_parameters():
                    p.data = (1 - self.lr_actor_target) * p.data + self.lr_actor_target * self.actor.state_dict()[name].data.clone()
                for name, p in self.critic_target.named_parameters():
                    p.data = (1 - self.lr_critic_target) * p.data + self.lr_critic_target * self.critic.state_dict()[name].data.clone()
                    
                
            self.train_step += 1

    # Output an action without noise for testing
    def take_action(self, state):
        with torch.no_grad():
            action = self.actor(torch.Tensor(state).cuda(device=self.device)).cpu().numpy()
            return action
        
    # Output a noised action for training.
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

    # Save model
    def save(self, episode):
        torch.save(self.actor.state_dict(), './DRL_Nets/actor_'+ str(episode) + '.pth')
        torch.save(self.actor_target.state_dict(), './DRL_Nets/actor_target_'+ str(episode) + '.pth')
        torch.save(self.critic.state_dict(), './DRL_Nets/critic_'+ str(episode) + '.pth')
        torch.save(self.critic_target.state_dict(), './DRL_Nets/critic_target_'+ str(episode) + '.pth')
        
    # Load model
    def load(self, episode):
        self.actor.load_state_dict(torch.load('./DRL_Nets/actor_'+ str(episode) + '.pth'))
        self.actor_target.load_state_dict(torch.load('./DRL_Nets/actor_target_'+ str(episode) + '.pth'))
        self.critic.load_state_dict(torch.load('./DRL_Nets/critic_'+ str(episode) + '.pth'))
        self.critic_target.load_state_dict(torch.load('./DRL_Nets/critic_target_'+ str(episode) + '.pth'))

# Training process for federated learning clients
class Training(mp.Process):
    def __init__(self,
                 C=4, # clipping bound
                 bs=batch_size, # batch size for local update
                 lr=0.1, # learning rate
                 el=epoch_local, # Number of epochs in a local update
                 ins=mp.Queue(), # Input queue for inter-process communication
                 outs=mp.Queue(), # Output queue for inter-process communication
                 device='cuda:0' # Selected GPU device
                 ) -> None:
        mp.Process.__init__(self)
        
        # Client parameters
        self.device = device
        self.C = C
        self.bs = bs
        self.el = el
        self.lr = lr
        self.trainloader = None
        self.trset = None
        
        # Processes communication
        self.ins = ins
        self.outs = outs
        
        
    # Run the process
    def run(self):
        # Initialize client dataset and network
        self.trset = self.ins.get()
        self.trainloader = torch.utils.data.DataLoader(self.trset, batch_size=self.bs,
                                          shuffle=True, num_workers=0, pin_memory=False)
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
        print('Dataset loaded successfully of', self.device)
        self.outs.put(1)
        
        # Select the action to perform based on the input mode: 0 - terminate process, 1 - train, or 2 - test
        while True:
            mode, net, sigma = self.ins.get()
            if mode == 0: # Terminate mode
                break
            elif mode == 1: # training mode
                self.net.load_state_dict(net.state_dict())
                self.optimizer.noise_multiplier = sigma
                for epoch in range(self.el):
                    for i, data in enumerate(self.trainloader):
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
                    for i, data in enumerate(self.trainloader):
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
                    
# 联邦学习Client
class Client():
    def __init__(self,
                 id=-1, # client id
                 C=4, # clipping bound
                 rho_total=20.0, # Total privacy budget for the client
                 bs=batch_size, # batch size
                 trset=None, # Client's local data set
                 lr=0.1, # Local update learning rate
                 el=epoch_local, # Number of epochs in a local update
                 hl=5, # Length of history (unused)
                 agent=None, # Client's DRL agent
                 device='cuda:0' # Selected GPU device
                 ) -> None:
        # Client information
        self.id = id
        self.rho = rho_total
        self.rho_total = rho_total
        self.C = C
        self.bs = bs
        self.trset = trset
        self.lr = lr
        self.el = el
        self.hl = hl
        self.sigma = sigma_max # Selected sigma
        self.rho_used = 2 * self.el / (self.sigma**2) # Sigma's corresponding privacy budget
        self.local_acc = 0.1 # Record the accuracy tested using a local dataset
        
        # DRL variables
        self.state = None
        self.action = None
        self.reward = None
        self.next_state = None
        self.done = None
        self.real_action = None
        
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
        
    # Reset client information other than dataset
    def reset(self, rho_total):
        self.rho = rho_total
        self.rho_total = rho_total
        self.action = None
        self.sigma = sigma_max
        self.local_acc = 0.1
        
    # Train
    def train(self, net, sigma):
        self.sigma = sigma
        rho_per_epoch = 2 / (sigma**2)
        self.rho_used = self.el * rho_per_epoch
        self.rho -= self.rho_used
        self.ins.put([1, net, sigma])
        
    # Test
    def test(self, net):
        self.ins.put([2, net, 0])
    
    # Terminate process
    def terminate_process(self):
        self.ins.put([0, 0, 0])
        
# Server
class Server():
    def __init__(self,
                 net, # Initialized global net
                 device='cuda:0' # Selected GPU device
                 ) -> None:
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

    # Aggregate the global model
    def aggregate(self, net_list):
        count = 0
        for net in net_list:
            self.dummy_net.load_state_dict(net.state_dict())
            for name, p in self.net.named_parameters():
                p.data = p.data * (count / (count + 1)) + self.dummy_net.state_dict()[name].data / (count + 1)
            count += 1
            
if __name__ == '__main__':
    t0 = time.time()
    agent = Agent(n_agents=N,
                  dim_state=2,
                  dim_action=1)
    mp.set_start_method('spawn', force=True)
    
    # Init server
    net = Net()
    server = Server(net=net)
    
    # Divide Datasets
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))])
    batch_size = 200
    trainsets = []
    for i in range(torch.cuda.device_count()):
        trainset = CUDAMNIST(root='./data', train=True,
                                        download=True, pre_transform=transform, device='cuda:' + str(i))
        trainsets.append(trainset)
    
    indices = list(range(len(trainsets[0])))
    random.shuffle(indices)
    dataset_num = int(len(trainsets[0]) / N)
    trsets = []
    for i in range(N):
        device_index = i % torch.cuda.device_count()
        device = 'cuda:' + str(device_index)
        trset = torch.utils.data.Subset(trainsets[device_index], indices[i * dataset_num: (i + 1) * dataset_num])
        trsets.append([trset, device])
    
    
    
    # Init clients
    clients = [Client(id=i, rho_total=np.random.uniform(low=rho_min, high=rho_max),
                    trset=trsets[i][0], agent=agent, device=trsets[i][1]) for i in range(N)]
    
    # Start DRL training ##################################################################################################################
    episode = 1
    final_accs = []
    mean_final_accs = []
    returns = []
    mean_returns = []
    high_acc = 0.0

    # resume from a previous run if a checkpoint exists (RESUME=0 disables)
    _r = resume_load(agent)
    if _r is not None:
        episode, final_accs, mean_final_accs, returns, mean_returns, high_acc = _r
    else:
        print('starting fresh (no checkpoint at %s)' % RESUME_PATH, flush=True)
    while episode < episodes:
        t1 = time.time()
        print('Trained steps', agent.train_step)
        print('Explore ', agent.explore)
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
            
        sigmas = [[] for _ in range(N)]
        rho_useds = [[] for _ in range(N)]
        
        # Test using the initial global model and record the accuracy
        global_accs = []
        for client in clients:
            client.test(server.net)
        last_global_loss = 0
        last_global_acc = 0
        for client in clients:
            acc, avg_loss = client.outs.get()
            last_global_loss += avg_loss
            last_global_acc += acc
        last_global_loss /= N
        last_global_acc /= N
        global_accs.append(last_global_acc)
        
        # Calculate the sigma for each client in each round using the DRL model
        isDone = 0.0
        for CR in range(CR_Total):
            if CR == CR_Total - 1:
                isDone = 1.0
            for i, client in enumerate(clients):
                state = [CR / CR_Total,
                         client.rho_total / rho_max]
                action, action_noise = agent.take_action_with_noise(state=state)
                sigma = action_noise[0] * (sigma_max - sigma_min) + sigma_min
                rho_used = sigma2rho(sigma=sigma, el=client.el, bs=client.bs)
                sigmas[i].append(sigma)
                rho_useds[i].append(rho_used)
        rho_used_sums = [sum(rho_useds[i]) for i in range(N)]
        
        # Correct sigma using the action mapping module
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
            
        # Federated learning using modified sigma
        isDone = 0.0
        for CR in range(CR_Total):
            if CR == CR_Total - 1:
                isDone = 1.0
            sigma_avg = 0
           
            for i, client in enumerate(clients):
                state = [CR / CR_Total,
                         client.rho_total / rho_max]
                client.state = state
                client.sigma = sigmas[i][CR]
                client.action = (sigmas[i][CR] - sigma_min) / (sigma_max - sigma_min)
                client.train(net=server.net, sigma=client.sigma)
            net_list = []
            for client in clients:
                cnet = client.outs.get()
                net_list.append(cnet)
            server.aggregate(net_list=net_list)
            for client in clients:
                client.test(server.net)
            
            global_loss = 0
            global_acc = 0
            for client in clients:
                acc, avg_loss = client.outs.get()
                global_loss += avg_loss
                global_acc += acc
                client.local_acc = acc
            global_loss /= N
            global_acc /= N
            
            states = [client.state for client in clients]
            actions = [client.action for client in clients]
            rewards = [global_acc - last_global_acc]
            global_accs.append(global_acc)
            last_global_acc = global_acc
            
            
            next_states = [[(CR + 1) / CR_Total,
                           client.rho_total / rho_max] for client in clients]
            done = [isDone]
            
            agent.train(state=states, action=actions, reward=rewards, next_state=next_states, done=done)
            
            
        # Tests are performed after each episode has been trained
        if episode % 1 == 0:
            net_init = Net()
            net = Net()
            net.load_state_dict(net_init.state_dict())
            server.reset(net=net)
            rhos_init = []
            
            for client in clients:
                rho = np.random.uniform(low=rho_min, high=rho_max)
                rhos_init.append(rho)
                
            for i, client in enumerate(clients):
                client.reset(rho_total=rhos_init[i])
                
            sigmas = [[] for _ in range(N)]
            rho_useds = [[] for _ in range(N)]
            
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
                
            print('Original sigmas:', [np.mean([sigmas[i][CR] for i in range(N)]) for CR in range(CR_Total)])
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
                    print('upper bound:', bub, 'lower bound:', blb, 'best b:', b)
                    if b > bub:
                        b = bub
                    elif b < blb:
                        b = blb
                    a = (client.rho_total - b * CR_Total) / rho_used_sums[i]
                for CR in range(CR_Total):
                    rho_useds[i][CR] = a * rho_useds[i][CR] + b
                    sigmas[i][CR] = rho2sigma(rho=rho_useds[i][CR], el=client.el, bs=client.bs)
                    
            for i, client in enumerate(clients):
                client.reset(rho_total=rhos_init[i])
            global_accs_DRL = []
            
            for client in clients:
                client.test(server.net)
            last_global_loss = 0
            last_global_acc = 0
            for client in clients:
                acc, avg_loss = client.outs.get()
                last_global_loss += avg_loss
                last_global_acc += acc
            last_global_loss /= N
            last_global_acc /= N
            global_accs_DRL.append(last_global_acc)
            
            for CR in range(CR_Total):
                for i, client in enumerate(clients):
                    sigma = sigmas[i][CR]
                    client.sigma = sigma
                    client.train(net=server.net, sigma=client.sigma)
                net_list = []
                for client in clients:
                    cnet = client.outs.get()
                    net_list.append(cnet)
                server.aggregate(net_list=net_list)
                for client in clients:
                    client.test(server.net)
                
                global_loss = 0
                global_acc = 0
                for client in clients:
                    acc, avg_loss = client.outs.get()
                    global_loss += avg_loss
                    global_acc += acc
                    client.local_acc = acc
                global_loss /= N
                global_acc /= N
                
                global_accs_DRL.append(global_acc)
                last_global_acc = global_acc
            returns_DRL = [global_accs_DRL[-1] - global_accs_DRL[0]]
            final_accs.append(global_accs_DRL[-1])
            returns.append(returns_DRL)
            
            low_index = max(len(final_accs) - 10, 0)
            mean_acc = np.mean(final_accs[low_index:])
            mean_final_accs.append(mean_acc)
            # Save the DRL model with high testing accuracy
            if mean_acc > high_acc and episode > 10:
                high_acc = mean_acc
                agent.save(episode=episode)
            
            low_index = max(len(returns) - 10, 0)
            mean_return = np.mean(returns[low_index:])
            mean_returns.append(mean_return)
        print('Time spent for this episode', time.time() - t1)
        
        episode += 1

        # crash-resume checkpoint, written atomically after the episode counter
        # advances so a resumed run restarts at the next unfinished episode
        if RESUME_ON and (episode % RESUME_EVERY == 0):
            resume_save(agent, episode, final_accs, mean_final_accs,
                        returns, mean_returns, high_acc)
            print('  [checkpoint saved: episode %d -> %s]' % (episode, RESUME_PATH),
                  flush=True)

        # periodic actor snapshot, named for the episode that produced it
        if SNAPSHOT_EVERY and ((episode - 1) % SNAPSHOT_EVERY == 0) and episode > 1:
            snapshot_save(agent, episode - 1)
            print('  [snapshot: actor @ episode %d -> %s]' % (episode - 1, SNAPSHOT_DIR),
                  flush=True)
            
    
    # Terminate all clients
    for client in clients:
        client.terminate_process()
    for client in clients:
        client.process.join()

    print('Time spend', time.time() - t0)