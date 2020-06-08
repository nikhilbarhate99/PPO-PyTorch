import os
from collections import deque
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym

from tensorboardX import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

    def __len__(self):
        return len(self.rewards)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax(dim=-1)
                )
        
        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 1)
                )
        
    def forward(self):
        raise NotImplementedError
        
    def act(self, state):
        state = torch.from_numpy(state).float().to(device) 
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.numpy(), log_prob
    
    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        state_value = self.value_layer(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy
        
class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, writer):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        self.writer = writer
    
    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.stack(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        
        # Optimize policy for K epochs:
        policy_losses = []
        value_losses = []
        entropy_losses = []
        losses = []

        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
                
            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            policy_loss = -torch.min(surr1, surr2)
            value_loss = 0.5*self.MseLoss(state_values, rewards)
            entropy_loss = - 0.01*dist_entropy
            loss = policy_loss + value_loss + entropy_loss
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            policy_losses.append(policy_loss.mean().item())
            value_losses.append(value_loss.mean().item())
            entropy_losses.append(entropy_loss.mean().item())
            losses.append(loss.mean().item())

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

        avg_policy_loss = np.mean(policy_losses)
        avg_value_loss = np.mean(value_losses)
        avg_entropy_loss = np.mean(entropy_losses)
        avg_loss = np.mean(losses)

        return avg_policy_loss, avg_value_loss, avg_entropy_loss, avg_loss
        

        
def main():
    ############## Hyperparameters ##############
    experiment_name = "ppo_original"
    env_name = "LunarLander-v2"
    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = 4
    render = False
    average_interval = 100
    solved_score = 230         # stop training if avg_reward > solved_reward
    max_episodes = 50000        # max training episodes
    max_timesteps = 300         # max timesteps in one episode
    n_latent_var = 64           # number of variables in hidden layer
    update_memory_size = 2000   # update policy when memory is maxed out
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    random_seed = None
    #############################################

    exp_dir = os.path.join("experiments", experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    writer = SummaryWriter(exp_dir)

    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, writer)

    # logging variables
    scores = deque(maxlen=average_interval)
    lengths = deque(maxlen=average_interval)

    # training loop
    for i_episode in range(1, max_episodes+1):
        episode_score = 0
        state = env.reset()
        for t in range(max_timesteps):

            # Running policy_old:
            with torch.no_grad():
                action, log_prob = ppo.policy_old.act(state)
            next_state, reward, done, _ = env.step(action)

            # update episode score
            episode_score += reward
            
            # Appending to the Memory as tensors
            memory.states.append(torch.from_numpy(state).float())
            memory.actions.append(torch.tensor(action).long())
            memory.logprobs.append(log_prob)
            memory.rewards.append(torch.tensor(reward).float())
            memory.is_terminals.append(done)
            
            # update if the memory is big enough
            memory_size = len(memory)
            if memory_size >= update_memory_size:
                # update ppo
                avg_policy_loss, avg_value_loss, avg_entropy_loss, avg_loss = ppo.update(memory)

                writer.add_scalar("info/avg_policy_loss", avg_policy_loss, i_episode)
                writer.add_scalar("info/avg_value_loss", avg_value_loss, i_episode)
                writer.add_scalar("info/avg_entropy_loss", avg_entropy_loss, i_episode)
                writer.add_scalar("info/avg_ppo_loss", avg_loss, i_episode)

                # clear memory
                memory.clear_memory()

            if render:
                env.render()

            state = next_state

            # if game is over
            if done:
                scores.append(episode_score)
                break

        # record play length
        lengths.append(t)
        
        # stop training if avg_reward > solved_reward
        avg_score = np.mean(scores)
        avg_length = np.mean(lengths)
        writer.add_scalar("info/avg_score", avg_score, i_episode)
        writer.add_scalar("info/avg_length", avg_length, i_episode)

        if avg_score > solved_score:
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './{}/PPO_{}.pth'.format(exp_dir, env_name))
            break
            
if __name__ == '__main__':
    main()
    
