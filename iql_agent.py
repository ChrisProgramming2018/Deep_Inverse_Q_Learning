import os
import sys
import numpy as np
import random
import gym
import gym.wrappers
from collections import namedtuple, deque
from models import QNetwork, Classifier
import torch
import torch.nn  as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
torch.set_printoptions(threshold=5000)
import logging
from datetime import datetime

logging.basicConfig(filename="test.log", level=logging.DEBUG)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class Agent():
    def __init__(self, state_size, action_size, config):
        self.env_name = config["env_name"]
        self.state_size = state_size
        self.action_size = action_size
        self.seed = 0
        self.device = 'cuda'
        print("cuda ", torch.cuda.is_available())
        self.lr_pre = config["lr_pre"]
        self.batch_size = config["batch_size"]
        self.lr = config["lr"]
        self.tau = config["tau"]
        self.gamma = 0.99
        self.qnetwork_local = QNetwork(state_size, action_size, self.seed).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, self.seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        self.soft_update(self.qnetwork_local, self.qnetwork_target, 1)
        
        self.q_shift_local = QNetwork(state_size, action_size, self.seed).to(self.device)
        self.q_shift_target = QNetwork(state_size, action_size, self.seed).to(self.device)
        self.optimizer_shift = optim.Adam(self.q_shift_local.parameters(), lr=self.lr)
        self.soft_update(self.q_shift_local, self.q_shift_target, 1)
         
        self.R_local = QNetwork(state_size, action_size, self.seed).to(self.device)
        self.R_target = QNetwork(state_size, action_size, self.seed).to(self.device)
        self.optimizer_r = optim.Adam(self.R_local.parameters(), lr=self.lr)
        self.soft_update(self.R_local, self.R_target, 1)
        
        self.memory = Memory(action_size, config["buffer_size"], self.batch_size, self.seed, self.device)
        self.t_step = 0
        self.steps = 0
        self.predicter = Classifier(state_size, action_size, self.seed).to(self.device)
        self.optimizer_pre = optim.Adam(self.predicter.parameters(), lr=self.lr_pre)
        pathname = "lr {} batch_size {} seed {}".format(self.lr, self.batch_size, self.seed)
        now = datetime.now()    
        dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
        pathname += dt_string
        tensorboard_name = str(config["locexp"]) + '/runs/' + pathname
        self.writer = SummaryWriter(tensorboard_name)
        print("summery writer ", tensorboard_name)
        self.average_prediction = deque(maxlen=100)
        self.average_same_action = deque(maxlen=100)
        self.all_actions = []
        for a in range(self.action_size):
            action = torch.Tensor(1) * 0 +  a
            self.all_actions.append(action.to(self.device))
    
    
    def learn(self, memory):
        states, next_states, actions, dones = memory.expert_policy(self.batch_size)
        self.steps += 1
        self.state_action_frq(states, actions)
        self.compute_shift_function(states, next_states, actions, dones)
        self.compute_r_function(states, actions)
        self.compute_q_function(states, next_states, actions, dones)
        return
    
    def learn_predicter(self, memory):
        """

        """
        states, next_states, actions, dones = memory.expert_policy(self.batch_size)
        self.state_action_frq(states, actions)
    
    def state_action_frq(self, states, action):
        """ Train classifer to compute state action freq
        """
        self.steps +=1
        #output = self.predicter(states.unsqueeze(0), train=True)
        
        self.predicter.train()
        output = self.predicter(states, train=True)
        output = output.squeeze(0)
        # logging.debug("out predicter {})".format(output))

        y = action.type(torch.long).squeeze(1)
        #print("y shape", y.shape)
        loss = nn.CrossEntropyLoss()(output, y)
        self.optimizer_pre.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.predicter.parameters(), 1)
        self.optimizer_pre.step()
        self.writer.add_scalar('Predict_loss', loss, self.steps)
        self.predicter.eval()

    def test_predicter(self, memory):
        """

        """
        self.predicter.eval()
        same_state_predition = 0
        for i in range(memory.idx):
            states = memory.obses[i]
            actions = memory.actions[i]
        
            states = torch.as_tensor(states, device=self.device).unsqueeze(0)
            actions = torch.as_tensor(actions, device=self.device)
            output = self.predicter(states)   
            output = F.softmax(output, dim=1)
            # create one hot encode y from actions
            y = actions.type(torch.long).item()
            p =torch.argmax(output.data).item()
            if y==p:
                same_state_predition += 1

        
        #self.average_prediction.append(same_state_predition)
        #average_pred = np.mean(self.average_prediction)
        #self.writer.add_scalar('Average prediction acc', average_pred, self.steps)
        #logging.debug("Same prediction {} of 100".format(same_state_predition))
        text = "Same prediction {} of {} ".format(same_state_predition, memory.idx)
        print(text)
        # self.writer.add_scalar('Action prediction acc', same_state_predition, self.steps)
        self.predicter.train()


    def get_action_prob(self, states, actions, predict=False):
        """
        """
        actions = actions.type(torch.long)
        # check if action prob is zero
        output = self.predicter(states)
        output = F.softmax(output, dim=1)
        # output = output.squeeze(0)
        action_prob = output.gather(1, actions)
        if predict:
            if action_prob.item() < 1e-3:
                return None
        action_prob = action_prob + torch.finfo(torch.float32).eps
        # logging.debug("action_prob {})".format(action_prob))
        action_prob = torch.log(action_prob)
        action_prob = torch.clamp(action_prob, min=-3, max=0)
        return action_prob

    def compute_shift_function(self, states, next_states, actions, dones):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        actions = actions.type(torch.int64)
        with torch.no_grad():
            # Get max predicted Q values (for next states) from target model
            Q_targets_next = self.qnetwork_target(next_states).max(1)[0].unsqueeze(1)
            # Compute Q targets for current states
            Q_targets = (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.q_shift_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer_shift.zero_grad()
        loss.backward()
        self.writer.add_scalar('Shift_loss', loss, self.steps)
        self.optimizer_shift.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.q_shift_local, self.q_shift_target)

    def compute_r_function(self, states, actions, debug=False):
        """

        """
        actions = actions.type(torch.int64)
        y = self.R_local(states).gather(1, actions)
        
        # sum all other actions
        # print("state shape ", states.shape)
        size = states.shape[0]
        idx = 0
        all_zeros = []
        with torch.no_grad():
            y_shift = self.q_shift_target(states).gather(1, actions)
            log_a = self.get_action_prob(states, actions)
            y_r_part1 = log_a - y_shift
            y_r_part2 =  torch.empty((size, 1), dtype=torch.float32).to(self.device)
            for a, s in zip(actions, states):
                y_h = 0
                taken_actions = 0
                for b in self.all_actions:
                    b = b.type(torch.int64).unsqueeze(1)
                    n_b = self.get_action_prob(s.unsqueeze(0), b, True)
                    if torch.eq(a, b) or n_b is None:
                        #print(n_b)
                        continue
                    taken_actions += 1
                    r_hat = self.R_target(s.unsqueeze(0)).gather(1, b)

                    y_s = self.q_shift_target(s.unsqueeze(0)).gather(1, b)
                    n_b = n_b - y_s

                    y_h += (r_hat - n_b)
                    if debug:
                        print("action", b.item())
                        print("r_pre {:.3f}".format(r_hat.item()))
                        print("n_b {:.3f}".format(n_b.item()))
                if taken_actions == 0:
                    y_r_part2[idx] = 0
                    y_r_part1[idx] = y[idx].detach()
                    all_zeros.append(idx)
                else:
                    y_r_part2[idx] = (1. / taken_actions)  * y_h
                idx += 1
            #print(y_r_part2, y_r_part1)
            y_r = y_r_part1 + y_r_part2
            #print("_________________")
            #print("r update zeros ", len(all_zeros))
        if debug:
            print("expet action ", actions.item())
            # print("y r {:.3f}".format(y.item()))
            # print("log a prob {:.3f}".format(log_a.item()))
            # print("n_a {:.3f}".format(y_r_part1.item()))
            print("Correct action p {:.3f} ".format(y.item()))
            print("Correct action target {:.3f} ".format(y_r.item()))
            print("part1 corret action {:.2f} ".format(y_r_part1.item()))
            print("part2 incorret action {:.2f} ".format(y_r_part2.item()))
        r_loss = F.mse_loss(y, y_r)
        #sys.exit()
        # Minimize the loss
        self.optimizer_r.zero_grad()
        r_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.R_local.parameters(), 5)
        self.optimizer_r.step()
        self.writer.add_scalar('Reward_loss', r_loss, self.steps)
        if debug:
            print("after update r pre ", self.R_local(states).gather(1, actions).item())
            print("after update r target ", self.R_target(states).gather(1, actions).item())
        # ------------------- update target network ------------------- #
        self.soft_update(self.R_local, self.R_target)
        if debug:
            print("after soft upda r target ", self.R_target(states).gather(1, actions).item())
    
    def compute_q_function(self, states, next_states, actions, dones, debug=False):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        actions = actions.type(torch.int64)
        if debug:
            print("---------------q_update------------------")
            print("expet action ", actions.item())
            print("state ", states)
        
        with torch.no_grad():
            # Get max predicted Q values (for next states) from target model
            Q_targets_next = self.qnetwork_target(next_states).max(1)[0].unsqueeze(1)
            # Compute Q targets for current states
            rewards = self.R_target(states).gather(1, actions)
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
            if debug:
                print("reward  {}".format(rewards.item()))
                print("Q target next {}".format(Q_targets_next.item()))
                print("Q_target {}".format(Q_targets.item()))



        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        if debug:
            print("q for a {}".format(Q_expected))
        #print("pre", Q_expected.shape)
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        self.writer.add_scalar('Q_loss', loss, self.steps)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if debug:
            print("q after update {}".format(self.qnetwork_local(states)))
            print("q loss {}".format(loss.item()))


        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)



    def dqn_train(self, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        env =  gym.make('LunarLander-v2')
        scores = []                        # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = eps_start
        for i_episode in range(1, n_episodes+1):
            state = env.reset()
            score = 0
            for t in range(max_t):
                self.t_step += 1
                action = self.dqn_act(state, eps)
                next_state, reward, done, _ = env.step(action)
                self.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    self.test_q()
                    break
            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            eps = max(eps_end, eps_decay*eps) # decrease epsilon
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if np.mean(scores_window)>=200.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
                break



    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.update_q(experiences)


    def dqn_act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def update_q(self, experiences, debug=False):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        # Get max predicted Q values (for next states) from target model
        with torch.no_grad():
            Q_targets_next = self.qnetwork_target(next_states).max(1)[0].unsqueeze(1)
            # Compute Q targets for current states
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        if debug:
            print("----------------------")
            print("----------------------")
            print("Q target", Q_targets)
            print("pre", Q_expected)
            print("all local",self.qnetwork_local(states))

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)



    def test_q(self):
        experiences = self.memory.test_sample()
        self.update_q(experiences, True)

    def test_q_value(self, memory):
        same_action = 0
        test_elements = memory.idx
        for i in range(test_elements):
            states = memory.obses[i]
            next_states = memory.next_obses[i]
            actions = memory.actions[i]
            dones = memory.not_dones[i]
            states = torch.as_tensor(states, device=self.device).unsqueeze(0)
            next_states = torch.as_tensor(next_states, device=self.device)
            actions = torch.as_tensor(actions, device=self.device)
            dones = torch.as_tensor(dones, device=self.device)
            with torch.no_grad():
                output = self.predicter(states, train=True)
                output = F.softmax(output, dim=1)
                q_values = self.qnetwork_local(states)
                best_action = torch.argmax(q_values).item()
                actions = actions.type(torch.int64)
                q_max = q_values.max(1)
            
            print("q values", q_values)
            print("q action", q_values[0][actions.item()].item())
            print("q best", q_max[0].data)

            if  actions.item() == best_action:
                same_action += 1
                continue
                print("-------------------------------------------------------------------------------")
                print("state ", i)
                print("expert ", actions)
                print("q values", q_values.data)
                print("action prob predicter  ", output.data)
                self.compute_r_function(states, actions.unsqueeze(0), True)
                self.compute_q_function(states, next_states.unsqueeze(0), actions.unsqueeze(0), dones, True)
            else:
                continue
                # logging.debug("experte action  {} q fun {}".format(actions.item(), q_values))
                print("-------------------------------------------------------------------------------")
                print("state ", i)
                print("expert ", actions)
                print("q values", q_values.data)
                print("action prob predicter  ", output.data)
                self.compute_r_function(states, actions.unsqueeze(0), True)
                self.compute_q_function(states, next_states.unsqueeze(0), actions.unsqueeze(0), dones, True)
             
        self.average_same_action.append(same_action)
        av_action = np.mean(self.average_same_action)
        self.writer.add_scalar('Same_action', same_action, self.steps)
        print("Same actions {}  of {}".format(same_action, test_elements))


    def soft_update(self, local_model, target_model, tau=None):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        if tau is None:
            tau = self.tau
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def save(self, filename):
        """

        """
        mkdir("", filename)
        torch.save(self.predicter.state_dict(), filename + "_predicter.pth")
        torch.save(self.optimizer_pre.state_dict(), filename + "_predicter_optimizer.pth")
        """
        torch.save(self.Q_local.state_dict(), filename + "_q_net.pth")
        torch.save(self.optimizer_q.state_dict(), filename + "_q_net_optimizer.pth")
        torch.save(self.q_shift_local.state_dict(), filename + "_q_shift_net.pth")
        torch.save(self.optimizer_q_shift.state_dict(), filename + "_q_shift_net_optimizer.pth")
        """
        print("save models to {}".format(filename))
    
    def load(self, filename):
        self.predicter.load_state_dict(torch.load(filename + "_predicter.pth"))
        self.optimizer_pre.load_state_dict(torch.load(filename + "_predicter_optimizer.pth"))
        print("Load models to {}".format(filename))
        

def mkdir(base, name):
    """
    Creates a direction if its not exist
    Args:
       param1(string): base first part of pathname
       param2(string): name second part of pathname
    Return: pathname
    """
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path






class Memory:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def test_sample(self):
        experiences = [self.memory[0]]
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        return (states, actions, rewards, next_states, dones)


    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        # print("ex", experiences)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)





