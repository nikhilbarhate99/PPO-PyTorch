import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(Model, self).__init__()
        self.affine = nn.Linear(state_dim, n_latent_var)
        
        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax()
                )
        
        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Linear(n_latent_var, 1)
                )
        
        # Memory:
        self.actions = []
        self.states = []
        self.logprobs = []
        self.state_values = []
        self.rewards = []
        
    def forward(self, state, action=None, evaluate=False):
        # if evaluate is True then we also need to pass an action for evaluation
        # else we return a new action from distribution
        if not evaluate:
            state = torch.from_numpy(state).float().to(device)
        
        state_value = self.value_layer(state)
        
        action_probs = self.action_layer(state)
        action_distribution = Categorical(action_probs)
        
        if not evaluate:
            action = action_distribution.sample()
            self.actions.append(action)
            
        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)
        
        if not evaluate:
            return action.item()
        
    def clearMemory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]
        
class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = Model(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                              lr=lr, betas=betas)
        self.policy_old = Model(state_dim, action_dim, n_latent_var).to(device)
        
        self.SmoothL1Loss = nn.SmoothL1Loss()
        
    def update(self):   
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward in reversed(self.policy_old.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-10)
        
        # convert list in tensor
        old_states = torch.tensor(self.policy_old.states).to(device).detach()
        old_actions = torch.tensor(self.policy_old.actions).to(device).detach()
        old_logprobs = torch.tensor(self.policy_old.logprobs).to(device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            self.policy(old_states, old_actions, evaluate=True)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            logprobs = self.policy.logprobs[0].to(device)
            ratios = torch.exp(logprobs - old_logprobs.detach())
                
            # Finding Surrogate Loss:
            state_values = self.policy.state_values[0].to(device)
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip)
            loss = -torch.min(surr1, surr2)
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
            self.policy.clearMemory()
        
        self.policy_old.clearMemory()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
def main():
    ############## Hyperparameters ##############
    env_name = "LunarLander-v2"
    #env_name = "CartPole-v1"
    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = 4
    
    render = False
    log_interval = 10
    n_latent_var = 64           # number of variables in hidden layer
    n_update = 10               # update policy every n episodes
    lr = 0.0008
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 5                # update policy for K epochs
    eps_clip = 0.1              # clip parameter for PPO
    random_seed = None
    #############################################
    
    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    print(lr,betas)
    
    running_reward = 0
    avg_length = 0
    for i_episode in range(1, 10000):
        state = env.reset()
        for t in range(10000):
            # Running policy_old:
            action = ppo.policy_old(state)
            state, reward, done, _ = env.step(action)
            
            # Saving state and reward:
            ppo.policy_old.states.append(state)
            ppo.policy_old.rewards.append(reward)
            
            running_reward += reward
            if render:
                env.render()
            if done:
                break
        
        avg_length += t
        
        # update after n episodes
        if i_episode % n_update:
            ppo.update()
        
        # log
        if running_reward > (log_interval*200):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), 
                       './LunarLander_{}_{}_{}.pth'.format(
                        lr, betas[0], betas[1]))
            break
        
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))
            
            print('Episode {} \t avg length: {} \t reward: {}'.format(
                    i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
            
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
