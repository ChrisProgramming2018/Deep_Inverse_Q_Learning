import os
import sys
import time
import numpy as np
import random
import gym
import gym.wrappers
from collections import namedtuple, deque
from models import QNetwork, Classifier, RNetwork, DQNetwork
import torch
import torch.nn  as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
torch.set_printoptions(threshold=5000)
import logging
from datetime import datetime
from utils import  mkdir


now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
mkdir("","search_results")
logging.basicConfig(filename="search_results/{}.log".format(dt_string), level=logging.DEBUG)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class Agent():
    def __init__(self, state_size, action_size, config):
        self.env_name = config["env_name"]
        self.state_size = state_size
        self.action_size = action_size
        self.seed = config["seed"]
        self.clip = config["clip"]
        self.device = 'cuda'
        print("Clip ", self.clip)
        print("cuda ", torch.cuda.is_available())
        self.double_dqn = config["DDQN"]
        print("Use double dqn", self.double_dqn)
        self.lr_pre = config["lr_pre"]
        self.batch_size = config["batch_size"]
        self.lr = config["lr"]
        self.tau = config["tau"]
        print("self tau", self.tau)
        self.gamma = 0.99
        self.fc1 = config["fc1_units"]
        self.fc2 = config["fc2_units"]
        self.fc3 = config["fc3_units"]
        self.qnetwork_local = QNetwork(state_size, action_size, self.fc1, self.fc2, self.fc3, self.seed).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, self.fc1, self.fc2,self.fc3,  self.seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        self.soft_update(self.qnetwork_local, self.qnetwork_target, 1)
        
        self.q_shift_local = QNetwork(state_size, action_size, self.fc1, self.fc2, self.fc3, self.seed).to(self.device)
        self.q_shift_target = QNetwork(state_size, action_size, self.fc1, self.fc2, self.fc3, self.seed).to(self.device)
        self.optimizer_shift = optim.Adam(self.q_shift_local.parameters(), lr=self.lr)
        self.soft_update(self.q_shift_local, self.q_shift_target, 1)
         
        self.R_local = QNetwork(state_size, action_size, self.fc1, self.fc2, self.fc3,  self.seed).to(self.device)
        self.R_target = QNetwork(state_size, action_size, self.fc1, self.fc2, self.fc3, self.seed).to(self.device)
        self.optimizer_r = optim.Adam(self.R_local.parameters(), lr=self.lr)
        self.soft_update(self.R_local, self.R_target, 1) 

        self.expert_q = DQNetwork(state_size, action_size, seed=self.seed).to(self.device)
        # self.expert_q.load_state_dict(torch.load('checkpoint.pth'))
        self.memory = Memory(action_size, config["buffer_size"], self.batch_size, self.seed, self.device)
        self.t_step = 0
        self.steps = 0
        # self.predicter = Classifier(state_size, action_size, self.seed).to(self.device)
        self.predicter = DQNetwork(state_size, action_size, self.seed).to(self.device)
        self.optimizer_pre = optim.Adam(self.predicter.parameters(), lr=self.lr_pre)
        pathname = "lr_{}_batch_size_{}_fc1_{}_fc2_{}_fc3_{}_seed_{}".format(self.lr, self.batch_size, self.fc1, self.fc2, self.fc3, self.seed)
        pathname += "_clip_{}".format(config["clip"])
        pathname += "_tau_{}".format(config["tau"])
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
    
    def find_idx_actions(self, states):
        list_idx = [1 for i in range(states.shape[0])]
        for idx, s in enumerate(states):
            for a in self.all_actions:
                n_a = self.get_action_prob(s.unsqueeze(0), a.unsqueeze(0)) 
                # if one action prob is to small set index to 0
                if n_a is None:
                    list_idx[idx] = 0
                    break
        return list_idx


    def learn(self, memory):
        # only update actions with action prob greater then 1e-4
        
        states, next_states, actions, dones = memory.expert_policy(self.batch_size)
        self.state_action_frq(states, actions)
        remove_idxs = self.find_idx_actions(states)
        for a in range(self.action_size):
                action =  torch.ones([self.batch_size, 1], device= self.device) * a
        # remove actions with less 1e-4 prob
        #for s, a in zip(states, actions):
        #    print(s, a)
        mask = torch.BoolTensor(remove_idxs)
        states = states[mask]
        next_states = next_states[mask]
        actions = action[mask]
        dones = dones[mask]
        print("batch size ", dones.shape)
        
        if dones.shape[0] <= 1:
            return
         
        self.steps += 1
        self.compute_shift_function(states, next_states, actions, dones)
        self.compute_r_function(states, actions)
        self.compute_q_function(states, next_states, actions, dones)
        self.soft_update(self.q_shift_local, self.q_shift_target, self.tau)
        self.soft_update(self.R_local, self.R_target, self.tau)
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
        return
    
    def learn_predicter(self, memory):
        """

        """
        states, next_states, actions, dones = memory.expert_policy(self.batch_size)
        self.state_action_frq(states, actions)
    
    def state_action_frq(self, states, action):
        """ Train classifer to compute state action freq
        """ 
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

        
        text = "Same prediction {} of {} ".format(same_state_predition, memory.idx)
        print(text)
        self.predicter.train()


    def get_action_prob(self, states, actions):
        """
        """
        self.predicter.eval()
        actions = actions.type(torch.long)
        # check if action prob is zero
        output = self.predicter(states.detach())
        output = F.softmax(output, dim=1)
        # print("get action_prob ", output) 
        # output = output.squeeze(0)
        action_prob = output.gather(1, actions.detach())
        action_prob = action_prob + torch.finfo(torch.float32).eps
        # check if one action if its to small
        if action_prob.shape[0] == 1:
            if action_prob.cpu().detach().numpy()[0][0] < 1e-4:
                return None
        # logging.debug("action_prob {})".format(action_prob))
        action_prob = torch.log(action_prob)
        action_prob = torch.clamp(action_prob, min= self.clip, max=0)
        self.predicter.train()
        return action_prob

    def compute_shift_function(self, states, next_states, actions, dones):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        actions = actions.type(torch.int64)
        # Get max predicted Q values (for next states) from target model
        if self.double_dqn:
            qt = self.q_shift_local(next_states)
            max_q, max_actions = qt.max(1)
            Q_targets_next = self.qnetwork_target(next_states).gather(1, max_actions.unsqueeze(1))
        else:
            Q_targets_next = self.qnetwork_target(next_states.detach()).detach().max(1)[0].unsqueeze(1)
            # Compute Q targets for current states
        Q_targets = (self.gamma * Q_targets_next * dones).detach()
        # Get expected Q values from local model
        Q_expected = self.q_shift_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer_shift.zero_grad()
        loss.backward()
        self.writer.add_scalar('Shift_loss', loss, self.steps)
        self.optimizer_shift.step()


    def compute_r_function(self, states, actions, debug=False, log=False):
        """

        """
        actions = actions.type(torch.int64)
        
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
                    n_b = self.get_action_prob(s.unsqueeze(0), b)
                    if torch.eq(a, b) or n_b is None:
                        #logging.debug("best action {} ".format(a))
                        #logging.debug("n_b action {} ".format(b))
                        #logging.debug("n_b {} ".format(n_b))
                        # print("action is None", n_b)
                        continue
                    taken_actions += 1
                    r = self.R_target(s.unsqueeze(0))
                    r_hat = self.R_target(s.unsqueeze(0)).gather(1, b)

                    y_s = self.q_shift_target(s.unsqueeze(0)).gather(1, b)
                    n_b = n_b - y_s

                    y_h += (r_hat - n_b)
                    if log:
                        logging.debug("r for s and b {}")
                        text = "r _hat {:.2f} - n_b  {:.2f} | sh {:.2f} ".format(r.item(), y_s.item(), n_b.item(), y_s2.item())
                        logging.debug(text)
                if taken_actions == 0:
                    all_zeros.append(idx)
                else:
                    y_r_part2[idx] = (1. / taken_actions)  * y_h
                idx += 1
            y_r = y_r_part1 + y_r_part2
        y = self.R_local(states).gather(1, actions)
        if log:
            text = "Action {:.2f}  y target {:.2f} =  n_a {:.2f} + {:.2f} and pre{:.2f}".format(actions.item(), y_r.item(), y_r_part1.item(), y_r_part2.item(), y.item())
            logging.debug(text)
            print("use log")
            return 

        if debug:
            print("expet action ", actions.item())
            print("Correct action p {:.3f} ".format(y.item()))
            print("Correct action target {:.3f} ".format(y_r.item()))
            print("part1 corret action {:.2f} ".format(y_r_part1.item()))
            print("part2 incorret action {:.2f} ".format(y_r_part2.item()))
        
        
        r_loss = F.mse_loss(y, y_r.detach())
        
        #sys.exit()
        # Minimize the loss
        self.optimizer_r.zero_grad()
        r_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.R_local.parameters(), 5)
        self.optimizer_r.step()
        self.writer.add_scalar('Reward_loss', r_loss, self.steps)
    
    def compute_q_function(self, states, next_states, actions, dones, debug=False, log= False):
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
        # Get max predicted Q values (for next states) from target model
        if self.double_dqn:
            qt = self.qnetwork_local(next_states)
            max_q, max_actions = qt.max(1)
            Q_targets_next = self.qnetwork_target(next_states).gather(1, max_actions.unsqueeze(1))
        else:
            Q_targets_next = self.qnetwork_target(next_states.detach()).max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        rewards = self.R_target(states.detach()).gather(1, actions.detach())
        Q_targets = rewards + (self.gamma * Q_targets_next * (dones)).detach()
        if log:
            logging.debug("reward tar  {}".format(rewards.item()))
            logging.debug("Q target next {}".format(Q_targets_next.item()))
            logging.debug("Q_targets {}".format(Q_targets.item()))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        # print(self.qnetwork_local(states))
        if log:
            logging.debug("Q pre {}".format(Q_expected.item()))
            text = "Action {:.2f}  q target {:.2f} =  r_a {:.2f} + target {:.2f} and pre{:.2f}".format(actions.item(), Q_targets.item(), rewards.item(), Q_targets_next.item(), Q_expected.item())
            logging.debug(text)
            print("q use log")
            return 
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        print(loss)
        self.writer.add_scalar('Q_loss', loss, self.steps)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()




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



    def test_policy(self, expert=False):
        env =  gym.make('LunarLander-v2')
        logging.debug("new episode")
        average_score = [] 
        average_steps = []
        average_action = []
        for i in range(5):
            state = env.reset()
            score = 0
            same_action = 0
            logging.debug("new episode")
            for t in range(200):
                state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                q_expert = self.expert_q(state)
                q_values = self.qnetwork_local(state)
                logging.debug("q expert a0: {:.2f} a1: {:.2f} a2: {:.2f} a3: {:.2f}".format(q_expert.data[0][0], q_expert.data[0][1], q_expert.data[0][2], q_expert.data[0][3]))
                logging.debug("q values a0: {:.2f} a1: {:.2f} a2: {:.2f} a3: {:.2f}  )".format(q_values.data[0][0], q_values.data[0][1], q_values.data[0][2], q_values.data[0][3]))
                action = torch.argmax(q_values).item()
                action_e = torch.argmax(q_expert).item()
                if action == action_e:
                    same_action += 1
                if expert:
                    action = action_e
                next_state, reward, done, _ = env.step(action)
                state = next_state
                score += reward
                if done:
                    average_score.append(score)
                    average_steps.append(t)
                    average_action.append(same_action)
                    break
        mean_steps = np.mean(average_steps)
        mean_score = np.mean(average_score)
        mean_action= np.mean(average_action)
        self.writer.add_scalar('Ave_epsiode_length', mean_steps , self.steps)
        self.writer.add_scalar('Ave_same_action', mean_action, self.steps)
        self.writer.add_scalar('Ave_score', mean_score, self.steps)


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
        test_elements = 100
        all_diff = 0
        error = True
        for i in range(test_elements):
            # print("lop", i)
            logging.debug("element {}".format(i))

            states = memory.obses[i]
            next_states = memory.next_obses[i]
            actions = memory.actions[i]
            dones = memory.not_dones[i]
            states = torch.as_tensor(states, device=self.device).unsqueeze(0)
            next_states = torch.as_tensor(next_states, device=self.device)
            actions = torch.as_tensor(actions, device=self.device)
            dones = torch.as_tensor(dones, device=self.device)
            logging.debug("state {}".format(states))
            with torch.no_grad():
                q_values = self.qnetwork_local(states.detach())
                expert_values = self.expert_q(states.detach())
                best_action = torch.argmax(q_values).item()
                actions = actions.type(torch.int64)
                q_max = q_values.max(1)
                
                q = q_values[0][actions.item()].item()
                max_q =  q_max[0].data.item()
                diff = max_q - q
                all_diff += diff
                r = self.R_local(states.detach())
                rt = self.R_target(states.detach())
                qt = self.qnetwork_target(states.detach())
                sh = self.q_shift_local(states.detach())
                self.predicter.eval()
                output = self.predicter(states.detach())
                output = F.softmax(output, dim=1)
                
                logging.debug("------------------action --------------------------------")
                logging.debug("expert action  {})".format(actions.item()))
                logging.debug("out predicter a0: {:.2f} a1: {:.2f} a2: {:.2f} a3: {:.2f}  )".format(output.data[0][0], output.data[0][1], output.data[0][2], output.data[0][3]))
                logging.debug("q values a0: {:.2f} a1: {:.2f} a2: {:.2f} a3: {:.2f}  )".format(q_values.data[0][0], q_values.data[0][1], q_values.data[0][2], q_values.data[0][3]))
                logging.debug("q shift  a0: {:.2f} a1: {:.2f} a2: {:.2f} a3: {:.2f}  )".format(sh.data[0][0], sh.data[0][1], sh.data[0][2], sh.data[0][3]))
                logging.debug("q target a0: {:.2f} a1: {:.2f} a2: {:.2f} a3: {:.2f}  )".format(qt.data[0][0], qt.data[0][1], qt.data[0][2], qt.data[0][3]))
                logging.debug("rewards a0: {:.2f} a1: {:.2f} a2: {:.2f} a3: {:.2f}  )".format(r.data[0][0], r.data[0][1], r.data[0][2], r.data[0][3]))
                logging.debug("re target a0: {:.2f} a1: {:.2f} a2: {:.2f} a3: {:.2f}  )".format(rt.data[0][0], rt.data[0][1], rt.data[0][2], rt.data[0][3]))
                logging.debug("---------Reward Function------------")
                # self.compute_r_function(states, actions, log= True)
                self.compute_q_function(states, next_states.unsqueeze(0), actions.unsqueeze(0), dones, log= True)
                """
                action = torch.Tensor(1) * 0 +  0
                self.compute_r_function(states, action.unsqueeze(0).to(self.device), log= True)
                action = torch.Tensor(1) * 0 +  1
                self.compute_r_function(states, action.unsqueeze(0).to(self.device), log= True)
                action = torch.Tensor(1) * 0 +  2
                self.compute_r_function(states, action.unsqueeze(0).to(self.device), log= True)
                action = torch.Tensor(1) * 0 +  3
                self.compute_r_function(states, action.unsqueeze(0).to(self.device), log= True)
                logging.debug("------------------Q Function --------------------------------")
                action = torch.Tensor(1) * 0 +  0
                self.compute_q_function(states, next_states.unsqueeze(0), action.unsqueeze(0).to(self.device), dones, log= True)
                action = torch.Tensor(1) * 0 +  1
                self.compute_q_function(states, next_states.unsqueeze(0), action.unsqueeze(0).to(self.device), dones, log= True)
                action = torch.Tensor(1) * 0 +  2
                self.compute_q_function(states, next_states.unsqueeze(0), action.unsqueeze(0).to(self.device), dones, log= True)
                action = torch.Tensor(1) * 0 +  3
                self.compute_q_function(states, next_states.unsqueeze(0), action.unsqueeze(0).to(self.device), dones, log= True)
                """ 
            if  actions.item() == best_action:
                same_action += 1
        
        self.writer.add_scalar('diff', all_diff, self.steps)
        self.average_same_action.append(same_action)
        av_action = np.mean(self.average_same_action)
        self.writer.add_scalar('Same_action', same_action, self.steps)
        print("Same actions {}  of {}".format(same_action, test_elements))
        self.predicter.train()


    def soft_update(self, local_model, target_model, tau=4):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def save(self, filename):
        """

        """
        mkdir("", filename)
        torch.save(self.predicter.state_dict(), filename + "_predicter.pth")
        torch.save(self.optimizer_pre.state_dict(), filename + "_predicter_optimizer.pth")
        torch.save(self.qnetwork_local.state_dict(), filename + "_q_net.pth")
        torch.save(self.R_local.state_dict(), filename + "_r_net.pth")
        """
        torch.save(self.optimizer_q.state_dict(), filename + "_q_net_optimizer.pth")
        torch.save(self.q_shift_local.state_dict(), filename + "_q_shift_net.pth")
        torch.save(self.optimizer_q_shift.state_dict(), filename + "_q_shift_net_optimizer.pth")
        """
        print("save models to {}".format(filename))
    
    def load(self, filename):
        self.predicter.load_state_dict(torch.load(filename + "_predicter.pth"))
        self.optimizer_pre.load_state_dict(torch.load(filename + "_predicter_optimizer.pth"))
        print("Load models to {}".format(filename))
        






def index_None_value(matrix):
    """
    Search tensor for None Value and return list of the index
    assume  
    """
    res = []
    numb_rows = matrix.shape[0]
    # print("check for none", matrix.shape)
    for idx in range(numb_rows):
        # print(matrix[idx])
        if matrix[idx] is None:
            res.append(idx)
    return res


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
