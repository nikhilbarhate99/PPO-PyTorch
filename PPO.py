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
        
        self.logprobs = []
        self.state_values = []
        self.rewards = []
        
    def forward(self, state):
        state = torch.from_numpy(state).float()
        state = self.affine(state)
        
        state_value = self.value_layer(state)
        
        action_probs = F.softmax(self.action_layer(state))
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()
        
        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)
        
        return action.item()
    
    def clearMemory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]
     
class PPO:
    def __init__(self, lr, betas):
        self.policy_target = Model() # Target policy
        self.optimizer_target = torch.optim.Adam(self.policy_target.parameters(), 
                                                 lr=lr, betas=betas)
        self.policy_old = Model() # Behavior policy
        self.optimizer_old = torch.optim.Adam(self.policy_old.parameters(),
                                              lr=lr, betas=betas)
        
    def update_old_policy(self, gamma=0.99):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward in self.policy_old.rewards[::-1]:
            discounted_reward = reward + (gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std())
        
        loss = 0
        for logprob, value, reward in zip(self.policy_old.logprobs, 
                                          self.policy_old.state_values, 
                                          rewards):
            advantage = reward - value.item()
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value, reward)
            loss += (action_loss + value_loss)
        
        self.optimizer_old.zero_grad()
        loss.backward()
        self.optimizer_old.step()
        self.policy_old.clearMemory()
        
    def update_target_policy(self, gamma=0.99, eps_clip=0.2):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward in self.policy_old.rewards[::-1]:
            discounted_reward = reward + (gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std())
        
        # Finding ratio the (prob_target/prob_old):
        ratios = []
        for logprob, logprob_old in zip(self.policy_target.logprobs, 
                                        self.policy_old.logprobs):
            ratios.append(torch.exp(logprob - logprob_old))
        
        loss = 0
        for ratio, value, reward in zip(ratios, self.policy_target.state_values,
                                        rewards):
            advantage = reward - value.item()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            action_loss = -torch.min(surr1, surr2)
            value_loss = F.smooth_l1_loss(value, reward)
            loss += (action_loss + value_loss)
        
        self.optimizer_target.zero_grad()
        loss.backward()
        self.optimizer_target.step()
        self.policy_target.clearMemory()
        self.policy_old.clearMemory()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy_target.state_dict())
        
def main():
    # Defaults parameters:
    #    gamma = 0.99
    #    lr = 0.02
    #    betas = (0.9, 0.999)
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
    
    ppo = PPO(lr, betas)
    print(lr,betas)
    running_reward = 0
    for i_episode in range(1, 10000):
        state = env.reset()
        if i_episode % 5 != 0:
            # Update old policy:
            for t in range(10000):
                action = ppo.policy_old(state)
                state, reward, done, _ = env.step(action)
                ppo.policy_old.rewards.append(reward)
                running_reward += reward
                if render and i_episode>1500:
                    env.render()
                if done:
                    break
            ppo.update_old_policy(gamma)
        
        else:
            # Update target policy:
            for t in range(10000):
                # Collecting logprobs and state values for target policy:
                ppo.policy_target(state)
                
                action = ppo.policy_old(state)
                state, reward, done, _ = env.step(action)
                ppo.policy_old.rewards.append(reward)
                running_reward += reward
                if render and i_episode>1500:
                    env.render()
                if done:
                    break
            ppo.update_target_policy(gamma, eps_clip)
            
        if running_reward > 4000:
            print("########## Solved! ##########")
            torch.save(ppo.policy_target.state_dict(), 
                       './preTrained/LunarLander_{}_{}_{}.pth'.format(
                        lr, betas[0], betas[1]))
            break
        if i_episode % 20 == 0:
            running_reward = running_reward/20
            print('Episode {}\tlength: {}\treward: {}'.format(i_episode, t, running_reward))
            running_reward = 0

if __name__ == '__main__':
    main()
    
    
    
    
