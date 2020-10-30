import os
import sys
import numpy as np
import random
from collections import namedtuple, deque
from models import QNetwork, RNetwork, PolicyNetwork, Classifier
import torch
import torch.nn  as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable


class Agent():
    def __init__(self, state_size, action_size, action_dim, config):
        self.state_size = state_size
        self.action_size = action_size
        self.action_dim = action_dim
        self.seed = 0
        self.device = 'cuda'
        self.batch_size = config["batch_size"]
        self.lr = 0.005
        self.gamma = 0.99
        self.q_shift_local = QNetwork(state_size, action_size, self.seed).to(self.device)
        self.q_shift_target = QNetwork(state_size, action_size, self.seed).to(self.device)
        self.Q_local = QNetwork(state_size, action_size, self.seed).to(self.device)
        self.Q_target = QNetwork(state_size, action_size, self.seed).to(self.device)
        self.R_local = RNetwork(state_size,action_size, self.seed).to(self.device)
        self.R_target = RNetwork(state_size, action_size, self.seed).to(self.device)
        self.predicter = Classifier(state_size, action_dim, self.seed).to(self.device)
        # optimizer
        self.optimizer_q_shift = optim.Adam(self.q_shift_local.parameters(), lr=self.lr)
        self.optimizer_q = optim.Adam(self.Q_local.parameters(), lr=self.lr)
        self.optimizer_r = optim.Adam(self.R_local.parameters(), lr=self.lr)
        self.optimizer_pre = optim.Adam(self.predicter.parameters(), lr=self.lr)
        pathname = "lr {} batch_size {} seed {}".format(self.lr, self.batch_size, self.seed)
        tensorboard_name = str(config["locexp"]) + '/runs/' + pathname 
        self.writer = SummaryWriter(tensorboard_name)
        self.average_prediction = deque(maxlen=100)
        self.average_same_action = deque(maxlen=100)
        self.steps = 0
        self.ratio = 1. / action_dim
        self.all_actions = []
        for a in range(self.action_dim):
            action = torch.Tensor(1) * 0 +  a
            self.all_actions.append(action.to(self.device))


    
    def learn(self, memory):
        states, next_states, actions = memory.expert_policy(self.batch_size)
        self.steps += 1
        # actions = actions[0]
        # print("states ",  states)
        # self.state_action_frq(states, actions)
        self.compute_shift_function(next_states, actions)
        self.compute_r_function(states, actions)
        self.compute_q_function(states, next_states, actions)
        # update local nets 
        self.soft_update(self.Q_local, self.Q_target)
        self.soft_update(self.q_shift_local, self.q_shift_target)
        self.soft_update(self.R_local, self.R_target)
        return


    def learn_predicter(self, memory):
        """
        
        """
        states, next_states, actions = memory.expert_policy(self.batch_size)
        self.state_action_frq(states, actions)

        
    def test_predicter(self, memory):
        """
        
        """
        same_state_predition = 0
        for i in range(100):
            states, next_states, actions = memory.expert_policy(1)
            output = self.predicter(states.unsqueeze(0))
            # create one hot encode y from actions
            y = actions.type(torch.long)[0][0].data
            p =torch.argmax(output.data).data
            if torch.equal(y,p):
                same_state_predition += 1
        self.average_prediction.append(same_state_predition)
        average_pred = np.mean(self.average_prediction)
        self.writer.add_scalar('Average prediction acc', average_pred, self.steps)

        print("Same prediction {} of 100".format(same_state_predition))
        #print("sum  out ", torch.sum(output))
        #print("compare pred {}  real {} ".format(output, y))
        #print("compare pred {}  real {} ".format(torch.argmax(output), y))

    def state_action_frq(self, states, action):
        """ Train classifer to compute state action freq
        """
        self.steps +=1
        output = self.predicter(states.unsqueeze(0))
        output = output.squeeze(0)
        #print("out shape", output.shape)
        # create one hot encode y from actions
        
        y = action.type(torch.long).squeeze(1)
        #print("y shape", y.shape)
        loss = nn.CrossEntropyLoss()(output, y)
        self.optimizer_pre.zero_grad()
        loss.backward()
        self.optimizer_pre.step()
        self.writer.add_scalar('Predict_loss', loss, self.steps)

    def get_action_prob(self, states, actions, dim=False):
        """

        """
        #actions = actions.type(torch.long).squeeze(1)
        #actions = actions.unsqueeze(0)
        # actions = actions.type(torch.long).view(-1)
        actions = actions.type(torch.long)
        
        # check if action prob is zero
        """
        if dim:
            output = self.predicter(states.unsqueeze(0))
            action_prob = output.gather(1, actions)
            action_prob = action_prob.detach() + torch.finfo(torch.float32).eps
            action_prob = torch.log(action_prob)
            return action_prob
        """
        output = self.predicter(states.unsqueeze(0))
        output = output.squeeze(0)
        # print("output shape ", output.shape)
        # print("action shape ", actions.shape)
        action_prob = output.gather(1, actions)
        action_prob = action_prob.detach() + torch.finfo(torch.float32).eps
        # print("action_prob ", action_prob.shape)
        action_prob = torch.log(action_prob)
        return action_prob

    def compute_q_function(self, states, next_states,  actions):
        """
        
        """
        actions = actions.type(torch.float)
        q_est = self.Q_local(states, actions)
        r_pre = self.R_target(states, actions)
        q_hat =  torch.empty((self.batch_size, 1), dtype=torch.float32).to(self.device)
        for idx, s in enumerate(next_states):
            q = []
            for action in self.all_actions:
                q.append(self.Q_target(s.unsqueeze(0), action.unsqueeze(0)))
            q_max = max(q)
            q_hat[idx]=  q_max
        q_hat *= self.gamma
        q_target = r_pre + q_hat
        q_loss = F.mse_loss(q_est, q_target)
        # Minimize the loss
        self.optimizer_q.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Q_local.parameters(), 1)
        self.optimizer_q.step()
        self.writer.add_scalar('Q_loss', q_loss, self.steps)
        #print("q update")




    def compute_shift_function(self, next_states,  actions):
        """
        
        """
        # compute difference between Q_shift and y_sh
        actions = actions.type(torch.float)
        q_sh_value = self.q_shift_local(next_states, actions)
        y_sh =  torch.empty((self.batch_size, 1), dtype=torch.float32).to(self.device)
        for idx, s in enumerate(next_states):
            q = []
            for action in self.all_actions:
                q.append(self.Q_target(s.unsqueeze(0), action.unsqueeze(0)))
            q_max = max(q)
            y_sh[idx]=  q_max
        y_sh *= self.gamma
        q_shift_loss = F.mse_loss(y_sh, q_sh_value)
        # Minimize the loss
        self.optimizer_q_shift.zero_grad()
        q_shift_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_shift_local.parameters(), 1)
        self.optimizer_q_shift.step()
        # print("q shift update")



    def compute_r_function(self, states, actions):
        """
        
        """
        actions = actions.type(torch.float)
        y = self.R_local(states, actions)
        y_shift = self.q_shift_target(states, actions)
        y_r_part1 = self.get_action_prob(states, actions) - y_shift
        #print("ratio ", self.ratio)
        # sum all other actions
        y_r_part2 =  torch.empty((self.batch_size, 1), dtype=torch.float32).to(self.device)
        idx = 0
        for a, s in zip(actions, states):
            y_h = 0
            for b in self.all_actions:
                if torch.eq(a, b):
                    continue
                #print("diff ac ", b)
                r_hat = self.R_target(s.unsqueeze(0), b.unsqueeze(0))
                n_b = self.get_action_prob(s.unsqueeze(0), b.unsqueeze(0), True) - self.q_shift_target(s.unsqueeze(0), b.unsqueeze(0))
                y_h += (r_hat - n_b)
            y_h = self.ratio * y_h
            y_r_part2[idx] = y_h
            idx += 1
        # print("shape of r y ", y.shape)
        # print("y r part 1 ", y_r_part1.shape)
        # print("y r part 2 ", y_r_part2.shape)
        y_r = y_r_part1 + y_r_part2
        r_loss = F.mse_loss(y, y_r)
        # Minimize the loss
        self.optimizer_r.zero_grad()
        r_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.R_local.parameters(), 1)
        self.optimizer_r.step()
        self.writer.add_scalar('Reward_loss', r_loss, self.steps)


    def soft_update(self, local_net, target_net, tau=1e-3):
        """ swaps the network weights from the online to the target
        Args:
           param1 (float): tau
        """
        for target_param, local_param in zip(target_net.parameters(), local_net.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau)*target_param.data)


    def act(self, state):
        state = torch.Tensor(state).to(self.device)
        qs = []
        for a in self.all_actions:
            qs.append(self.Q_local(state.unsqueeze(0), a.unsqueeze(0)))

        return np.argmax(qs)

    def eval_policy(self, env, episode=2):
        scores = 0
        for i_episode in range(episode):
            score = 0
            state = env.reset()
            while True:
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                score += reward
                env.render()
                if done:
                    scores += score
                    break

        env.close()
        scores /= episode
        print("Average score {}".format(scores))



    def test_q_value(self, memory):
        same_action = 0
        for i in range(100):
            states, next_states, actions = memory.expert_policy(1)
            q_values = []
            for a in self.all_actions:
                q_values.append(self.Q_local(states, a.unsqueeze(0)))
            best_action = np.argmax(q_values)
            if  actions[0][0].item() == best_action:
                same_action += 1
        print("    ")
        print("Q values a0: {:.2f} a1: {:.2f} a2: {:.2f} a3: {:.2f}".format(q_values[0][0][0].data, q_values[1][0][0].data, q_values[2][0][0].data, q_values[2][0][0].data))
        print("same action {} of 100".format(same_action))
        self.average_same_action.append(same_action)
        av_action = np.mean(self.average_same_action)
        self.writer.add_scalar('Average_same_action', av_action, self.steps)

    def save(self, filename):
        """
        
        """
        mkdir("", filename)
        torch.save(self.predicter.state_dict(), filename + "_predicter.pth")
        torch.save(self.optimizer_pre.state_dict(), filename + "_predicter_optimizer.pth")
        print("save models to {}".format(filename))
    
    
    
    def load(self, filename):
        """
        
        """
        self.predicter.load_state_dict(torch.load(filename + "_predicter.pth"))
        self.optimizer_pre.load_state_dict(torch.load(filename + "_predicter_optimizer.pth"))


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



