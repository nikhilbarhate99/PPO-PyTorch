import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gym

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.affine = nn.Linear(8, 128)
        
        self.action_layer = nn.Linear(128, 4)
        self.value_layer = nn.Linear(128, 1)
        
        # Memory :
        self.actions = []
        self.states = []
        self.logprobs = []
        self.state_values = []
        self.rewards = []
        
    def forward(self):
        raise NotImplementedError
        
    def act(self, state): 

        state = torch.from_numpy(state).float()
        state = self.affine(state)
        
        state_value = self.value_layer(state)
        
        action_probs = F.softmax(self.action_layer(state))
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()
        
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)
        
        return action.item()   
    
    def evaluateAction(self, state, action):
        action_probs = F.softmax(self.action_layer(state))
        action_distribution = Categorical(action_probs)
        logprob = action_distribution.log_prob(action)
        self.logprobs.append(logprob)
         
    def clearMemory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]
        
class Experience:
    def __init__(self):
        # Memory :
        self.actions = []
        self.states = []
        self.logprobs = []
        self.state_values = []
        self.rewards = []
    
    def backupExperience(self, policy):
        for action, state, logprob, value, reward in zip(
                policy.actions, policy.states, policy.logprobs,
                policy.state_values, policy.rewards):
            
            self.actions.append(action)
            self.states.append(state)
            self.logprobs.append(logprob)
            self.state_values.append(value)
            self.rewards.append(reward)
        
class PPO:
    def __init__(self, lr, betas, gamma, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        
        self.policy = Model()
        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                              lr=lr, betas=betas)
        
    def update(self):
        exp_old = Experience()
        exp_old.backupExperience(self.policy)
        
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward in self.policy.rewards[::-1]:
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5 )
        
        # Updating the policy_theta_old to policy_theta :
        loss_old = 0
        for logprob, value, reward in zip(self.policy.logprobs, 
                                          self.policy.state_values, 
                                          rewards):
            advantage = reward - value.item()
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value, reward)
            loss_old += (action_loss + value_loss)
        
        self.optimizer.zero_grad()
        loss_old.backward(retain_graph=True)
        self.optimizer.step()
        self.policy.clearMemory()
        
        # Evaluating old actions :
        for state, action in zip(exp_old.states, exp_old.actions):
            self.policy.evaluateAction(state, action)
            

        # Finding the ratio (prob_target/prob_old):
        ratios = []
        for logprob, logprob_old in zip(self.policy.logprobs, 
                                        exp_old.logprobs):
            ratios.append(torch.exp(logprob - logprob_old))
        
        # Finding Surrogate Loss:
        loss = 0
        for ratio, value, reward in zip(ratios, exp_old.state_values,
                                        rewards):
            advantage = reward - value.item()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
            action_loss = -torch.min(surr1, surr2)
            value_loss = F.smooth_l1_loss(value, reward)
            loss += (action_loss + value_loss)
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.policy.clearMemory()
        
        # Copy new weights into old policy:
        #self.policy_old.load_state_dict(self.policy.state_dict())
        
def main():
    # Good parameters:
    #    gamma = 0.99
    #    lr = 0.01
    #    betas = (0.9, 0.999)
    #    eps_clip = 0.2
    #    random_seed = 543
    
    render = False
    gamma = 0.99
    lr = 0.02
    betas = (0.9, 0.999)
    eps_clip = 0.2
    random_seed = 543
    torch.manual_seed(random_seed)
    
    env = gym.make('LunarLander-v2')
    env.seed(random_seed)
    
    ppo = PPO(lr, betas, gamma, eps_clip)
    print(lr,betas)
    
    running_reward = 0
    avg_length = 0
    for i_episode in range(1, 10000):
        state = env.reset()
        for t in range(10000):
            # Run policy:
            action = ppo.policy.act(state)
            state, reward, done, _ = env.step(action)
            ppo.policy.rewards.append(reward)
            running_reward += reward
            if render and i_episode>1500:
                env.render()
            if done:
                break
        ppo.update()
        avg_length += t
        
        if running_reward > 4000:
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), 
                       './preTrained/LunarLander_{}_{}_{}.pth'.format(
                        lr, betas[0], betas[1]))
            break
        
        if i_episode % 20 == 0:
            avg_length = int(avg_length/20)
            running_reward = int((running_reward/20))
                
            print('Episode {}\tlength: {}\treward: {}'.format(
                    i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
            
if __name__ == '__main__':
    main()
    
    
