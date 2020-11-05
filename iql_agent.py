import os
import sys
import numpy as np
import random
import gym
import gym.wrappers
from collections import namedtuple, deque
from models import QNetwork, RNetwork, Classifier
import torch
import torch.nn  as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
torch.set_printoptions(threshold=5000)
import logging
 
 
logging.basicConfig(filename="test.log", level=logging.DEBUG)




class Agent():
    def __init__(self, state_size, action_size, action_dim, config):
        self.env_name = config["env_name"]
        self.state_size = state_size
        self.action_size = action_size
        self.action_dim = action_dim
        self.seed = 0
        self.device = 'cuda'
        print("cuda ", torch.cuda.is_available())
        self.batch_size = config["batch_size"]
        self.lr = config["lr"]
        self.lr_pre = config["lr_pre"]
        self.gamma = 0.99
        self.q_shift_local = QNetwork(state_size, action_dim, self.seed).to(self.device)
        self.q_shift_target = QNetwork(state_size, action_dim, self.seed).to(self.device)
        self.Q_local = QNetwork(state_size, action_dim, self.seed).to(self.device)
        self.Q_target = QNetwork(state_size, action_dim, self.seed).to(self.device)
        self.R_local = RNetwork(state_size,action_dim, self.seed).to(self.device)
        self.R_target = RNetwork(state_size, action_dim, self.seed).to(self.device)
        self.predicter = Classifier(state_size, action_dim, self.seed).to(self.device)
        self.soft_update(self.Q_local, self.Q_target, 1)
        self.soft_update(self.q_shift_local, self.q_shift_target, 1)
        self.soft_update(self.R_local, self.R_target, 1)
         
        # optimizer
        self.optimizer_q_shift = optim.Adam(self.q_shift_local.parameters(), lr=self.lr)
        self.optimizer_q = optim.Adam(self.Q_local.parameters(), lr=self.lr)
        self.optimizer_r = optim.Adam(self.R_local.parameters(), lr=0.001)
        self.optimizer_pre = optim.Adam(self.predicter.parameters(), lr=self.lr_pre)    
        pathname = "lr {} batch_size {} seed {}".format(self.lr, self.batch_size, self.seed)
        tensorboard_name = str(config["locexp"]) + '/runs/' + pathname 
        self.writer = SummaryWriter(tensorboard_name)
        print("summery writer ", tensorboard_name)
        self.average_prediction = deque(maxlen=100)
        self.average_same_action = deque(maxlen=100)
        self.steps = 0
        self.a0 = 0
        self.a1 = 1
        self.a2 = 2
        self.a3 = 3
        self.ratio = 1. / (action_dim - 1)
        self.all_actions = []
        for a in range(self.action_dim):
            action = torch.Tensor(1) * 0 +  a
            self.all_actions.append(action.to(self.device))
        self.debug_max = 10
    def debug(self, actions):

        if actions is None:
            al = float(self.a0 + self.a1 + self.a2 + self.a3)
            return [self.a0 /al , self.a1 /al, self.a2/al, self.a3/al]
        if actions == 0:
            self.a0 += 1
        if actions == 1:
            self.a1 += 1
        if actions == 2:
            self.a2 += 1
        if actions == 3:
            self.a3 += 1

    def learn(self, memory):
        states, next_states, actions, dones = memory.expert_policy(self.batch_size)
        self.steps += 1
        for i in range(3):
            self.state_action_frq(states, actions)
        self.compute_shift_function(states, next_states, actions)
        self.compute_r_function(states, actions)
        self.compute_q_function(states, next_states, actions, dones)
        # update local nets 
        self.soft_update(self.Q_local, self.Q_target)
        self.soft_update(self.q_shift_local, self.q_shift_target)
        self.soft_update(self.R_local, self.R_target)
        return


    def learn_predicter(self, memory):
        """
        
        """
        states, next_states, actions, dones = memory.expert_policy(self.batch_size)
        self.state_action_frq(states, actions)

        
    def test_predicter(self, memory):
        """
        
        """
        self.predicter.eval()
        same_state_predition = 0
        for i in range(5):
            states, next_states, actions, done = memory.expert_policy(1)
            output = self.predicter(states)
            output = F.softmax(output, dim=1)
            # create one hot encode y from actions
            y = actions.type(torch.long)[0][0].data
            p =torch.argmax(output.data).data
            if torch.equal(y,p):
                same_state_predition += 1
                print("expert ", y)
                print("q ", output.data)
        self.average_prediction.append(same_state_predition)
        average_pred = np.mean(self.average_prediction)
        self.writer.add_scalar('Average prediction acc', average_pred, self.steps)
        logging.debug("Same prediction {} of 100".format(same_state_predition))
        print("Same prediction {} of 100".format(same_state_predition))
        self.predicter.train()

    def state_action_frq(self, states, action):
        """ Train classifer to compute state action freq
        """
        self.steps +=1
        #output = self.predicter(states.unsqueeze(0), train=True)
        self.predicter.eval()
        output = self.predicter(states, train=True)
        output = output.squeeze(0)
        logging.debug("out predicter {})".format(output))
        
        y = action.type(torch.long).squeeze(1)
        #print("y shape", y.shape)
        loss = nn.CrossEntropyLoss()(output, y)
        self.optimizer_pre.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predicter.parameters(), 1)
        self.optimizer_pre.step()
        self.writer.add_scalar('Predict_loss', loss, self.steps)

    def get_action_prob(self, states, actions):
        """

        """
        actions = actions.type(torch.long)
        # check if action prob is zero
        output = self.predicter(states)
        output = F.softmax(output, dim=1)
        # output = output.squeeze(0)
        action_prob = output.gather(1, actions)
        # action_prob = action_prob.detach() + torch.finfo(torch.float32).eps
        logging.debug("action_prob {})".format(action_prob))
        action_prob = torch.log(action_prob)
        return action_prob

    def compute_q_function(self, states, next_states,  actions, dones, debug=False):
        """
        
        """
        self.Q_local.train()
        #print("ac", actions.shape)
        #print("s", states.shape)
        #print("ns", next_states.shape)
        actions = actions.type(torch.int64)
        q_est = self.Q_local(states).gather(1, actions).squeeze(1)
        r_pre = self.R_target(states).gather(1, actions).squeeze(1)
        target_action = self.Q_local(next_states)
        #print(target_action)
        target_action = torch.argmax(target_action,dim=1)
        #print(target_action)
        Q_target_next = self.Q_target(next_states).gather(1, target_action.unsqueeze(0)).squeeze(0)
        #print("Q target", Q_target_next.shape)
        #print("r_pre ", r_pre.shape)
        target_Q = r_pre + (self.gamma * Q_target_next) 
        if debug:
            print("---------------q_update------------------")
            print("expet action ", actions.item())
            print("q est {}".format(self.Q_local(states)))
            print("q for a {}".format(q_est))
            print("re  est {}".format( self.R_target(states)))
            print("re for a {}".format(r_pre))
            print("q next {}".format(self.Q_target(next_states)))
            print("q target {}".format(target_Q))
        
        #print("final target shape", target_Q.shape)
        #print("q est", q_est.shape)
        #print("q tar", target_Q.shape)
        q_loss = F.mse_loss(q_est, target_Q)

        # Minimize the loss
        self.optimizer_q.zero_grad()
        q_loss.backward()
        # print("q_loss ", q_loss)
        torch.nn.utils.clip_grad_norm_(self.Q_local.parameters(), 1)
        self.optimizer_q.step()
        self.writer.add_scalar('Q_loss', q_loss, self.steps)
        self.Q_local.eval()




    def compute_shift_function(self, states, next_states,  actions):
        """
        
        """
        # compute difference between Q_shift and y_sh
        
        actions = actions.type(torch.int64)
        q_sh_value = self.q_shift_local(states).gather(1, actions).squeeze(1)
        
        target_action = self.Q_local(next_states)
        target_action = torch.argmax(target_action,dim=1)
        shift_q_target_ = self.Q_target(next_states).gather(1, target_action.unsqueeze(0)).squeeze(0)
          
        shift_q_target_ = self.gamma * shift_q_target_
        q_shift_loss = F.mse_loss(q_sh_value, shift_q_target_)
        
        # Minimize the loss
        self.optimizer_q_shift.zero_grad()
        q_shift_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_shift_local.parameters(), 1)
        self.optimizer_q_shift.step()



    def compute_r_function(self, states, actions, debug=False):
        """
        
        """
        actions = actions.type(torch.int64)
        #y = self.R_local(states).gather(1, actions).squeeze(1).unsqueeze(1)
        y = self.R_local(states).gather(1, actions)
        y_shift = self.q_shift_target(states).gather(1, actions)
        log_a = self.get_action_prob(states, actions)

        y_r_part1 = log_a - y_shift 
        if debug:
            print("expet action ", actions.item())
            print("y r {:.3f}".format(y.item()))
            print("log a prob {:.3f}".format(log_a.item()))
            print("n_a {:.3f}".format(y_r_part1.item()))
        # sum all other actions
        # print("state shape ", states.shape)
        size = states.shape[0]
        y_r_part2 =  torch.empty((size, 1), dtype=torch.float32).to(self.device)
        idx = 0
        for a, s in zip(actions, states):
            y_h = 0
            for b in self.all_actions:
                if torch.eq(a, b):
                    continue
                b = b.type(torch.int64).unsqueeze(1)
                r_hat = self.R_target(s.unsqueeze(0)).gather(1, b)
                 
                y_s = self.q_shift_target(s.unsqueeze(0)).gather(1, b)
                n_b = self.get_action_prob(s.unsqueeze(0), b) - y_s
                if debug:
                    n_b = self.get_action_prob(s.unsqueeze(0), b)  # debuging
                    n_b = n_b - y_s 
                
                y_h += (r_hat - n_b)
                if debug:
                    print("action", b.item())
                    print("r_pre {:.3f}".format(r_hat.item()))
                    print("n_b {:.3f}".format(n_b.item()))
            y_r_part2[idx] = self.ratio * y_h
            idx += 1
        y_r = y_r_part1 + y_r_part2
        if debug:
            print("Correct action p {:.3f} ".format(y.item()))
            print("Correct action target {:.3f} ".format(y_r.item()))
            #print("part incorret action {:.2f} ".format(y_r_part2.item()))
        r_loss = F.mse_loss(y, y_r)
        
        # Minimize the loss
        self.optimizer_r.zero_grad()
        r_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.R_local.parameters(), 5)
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
        action =torch.argmax(self.Q_local(state.unsqueeze(0)))
        return action.item()

    def eval_policy(self, env, episode=2):
        pathname = "videos/time_steps-{}/".format(self.steps)
        mkdir("", pathname)
        env = gym.make(self.env_name)
        env.metadata["render.modes"] = ["human", "rgb_array"]
        env = gym.wrappers.Monitor(env=env, directory=pathname, force=True)
        scores = 0
        for i_episode in range(episode):
            score = 0
            state = env.reset()
            while True:
                action = self.act(state)
                state = torch.Tensor(state).to(self.device)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                score += reward
                env.render()
                if done:
                    scores += score
                    break

        env.close()
        scores /= episode
        print("Average score {}".format(scores))
        self.writer.add_scalar('Eval_policy', scores, self.steps)



    def test_q_value(self, memory):
        same_action = 0
        for i in range(self.debug_max):
            states = memory.obses[i]
            next_states = memory.next_obses[i]
            actions = memory.actions[i]
            dones = memory.not_dones[i]
            states = torch.as_tensor(states, device=self.device).unsqueeze(0)
            next_states = torch.as_tensor(next_states, device=self.device)
            actions = torch.as_tensor(actions, device=self.device)
            dones = torch.as_tensor(dones, device=self.device)
            output = self.predicter(states, train=True)
            output = F.softmax(output, dim=1)
            q_values = self.Q_local(states)
            best_action = torch.argmax(q_values).item()
            self.debug(best_action)
            if  actions.item() == best_action:
                same_action += 1
            else:
                logging.debug("experte action  {} q fun {}".format(actions.item(), q_values))
                print("-------------------------------------------------------------------------------")
                print("expert ", actions)
                print("q", q_values.data)
                print("a ", output.data)
                self.compute_r_function(states, actions.unsqueeze(0), True) 
                self.compute_q_function(states, next_states.unsqueeze(0), actions.unsqueeze(0), dones, True) 
        #al = self.debug(None)
        #print("inverse action a0: {:.2f} a1: {:.2f} a2: {:.2f} a3: {:.2f}".format(al[0], al[1], al[2], al[3]))
        print("same action {} of {}".format(same_action, self.debug_max))
        self.average_same_action.append(same_action)
        av_action = np.mean(self.average_same_action)
        self.writer.add_scalar('Average_same_action', av_action, self.steps)

    def save(self, filename):
        """
        
        """
        mkdir("", filename)
        torch.save(self.predicter.state_dict(), filename + "_predicter.pth")
        torch.save(self.optimizer_pre.state_dict(), filename + "_predicter_optimizer.pth")
        torch.save(self.Q_local.state_dict(), filename + "_q_net.pth")
        torch.save(self.optimizer_q.state_dict(), filename + "_q_net_optimizer.pth")
        torch.save(self.q_shift_local.state_dict(), filename + "_q_shift_net.pth")
        torch.save(self.optimizer_q_shift.state_dict(), filename + "_q_shift_net_optimizer.pth")
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



