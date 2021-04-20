## Hyperparameters

Hyperparameters used to obtain the `preTrained` networks are listed below :


### RoboschoolHalfCheetah-v1

```
####### initialize environment hyperparameters ######

env_name = "RoboschoolHalfCheetah-v1"

has_continuous_action_space = True

max_ep_len = 1000                   # max timesteps in one episode
max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

print_freq = max_ep_len * 10               # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2                 # log avg reward in the interval (in num timesteps)
save_model_freq = int(1e5)      # save model frequency (in num timesteps)

action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)

#####################################################


## Note : print/log frequencies should be > than max_ep_len


################ PPO hyperparameters ################

update_timestep = max_ep_len * 4      # update policy every n timesteps
K_epochs = 80               # update policy for K epochs in one PPO update

eps_clip = 0.2          # clip parameter for PPO
gamma = 0.99            # discount factor

lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network

random_seed = 0         # set random seed if required (0 = no random seed)

#####################################################
```

### RoboschoolHopper-v1

```
####### initialize environment hyperparameters ######

env_name = "RoboschoolHopper-v1"

has_continuous_action_space = True

max_ep_len = 1000                   # max timesteps in one episode
max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

print_freq = max_ep_len * 10               # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2                 # log avg reward in the interval (in num timesteps)
save_model_freq = int(1e5)      # save model frequency (in num timesteps)

action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)

#####################################################


## Note : print/log frequencies should be > than max_ep_len


################ PPO hyperparameters ################

update_timestep = max_ep_len * 4      # update policy every n timesteps
K_epochs = 80               # update policy for K epochs in one PPO update

eps_clip = 0.2          # clip parameter for PPO
gamma = 0.99            # discount factor

lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network

random_seed = 0         # set random seed if required (0 = no random seed)

#####################################################
```

### RoboschoolWalker2d-v1

```
####### initialize environment hyperparameters ######

env_name = "RoboschoolWalker2d-v1"

has_continuous_action_space = True

max_ep_len = 1000                   # max timesteps in one episode
max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

print_freq = max_ep_len * 10               # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2                 # log avg reward in the interval (in num timesteps)
save_model_freq = int(1e5)      # save model frequency (in num timesteps)

action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)

#####################################################


## Note : print/log frequencies should be > than max_ep_len


################ PPO hyperparameters ################

update_timestep = max_ep_len * 4      # update policy every n timesteps
K_epochs = 80               # update policy for K epochs in one PPO update

eps_clip = 0.2          # clip parameter for PPO
gamma = 0.99            # discount factor

lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network

random_seed = 0         # set random seed if required (0 = no random seed)

#####################################################
```


### BipedalWalker-v2

```
####### initialize environment hyperparameters ######

env_name = "BipedalWalker-v2"

has_continuous_action_space = True

max_ep_len = 1500                   # max timesteps in one episode
max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

print_freq = max_ep_len * 4               # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2                 # log avg reward in the interval (in num timesteps)
save_model_freq = int(1e5)      # save model frequency (in num timesteps)

action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)

#####################################################


## Note : print/log frequencies should be > than max_ep_len


################ PPO hyperparameters ################

update_timestep = max_ep_len * 4      # update policy every n timesteps
K_epochs = 80               # update policy for K epochs in one PPO update

eps_clip = 0.2          # clip parameter for PPO
gamma = 0.99            # discount factor

lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network

random_seed = 0         # set random seed if required (0 = no random seed)

#####################################################
```


### Cartpole-v1

```
####### initialize environment hyperparameters ######

env_name = "CartPole-v1"
has_continuous_action_space = False

max_ep_len = 400                    # max timesteps in one episode
max_training_timesteps = int(1e5)   # break training loop if timeteps > max_training_timesteps

print_freq = max_ep_len * 4     # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2       # log avg reward in the interval (in num timesteps)
save_model_freq = int(2e4)      # save model frequency (in num timesteps)

action_std = None


#####################################################


## Note : print/log frequencies should be > than max_ep_len


################ PPO hyperparameters ################


update_timestep = max_ep_len * 4      # update policy every n timesteps
K_epochs = 40               # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
gamma = 0.99                # discount factor

lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network

random_seed = 0         # set random seed if required (0 = no random seed)

#####################################################
```


### LunarLander-v2

```
####### initialize environment hyperparameters ######

env_name = "LunarLander-v2"
has_continuous_action_space = False

max_ep_len = 300                   # max timesteps in one episode
max_training_timesteps = int(1e6)   # break training loop if timeteps > max_training_timesteps

print_freq = max_ep_len * 8                # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2                  # log avg reward in the interval (in num timesteps)
save_model_freq = int(5e4)      # save model frequency (in num timesteps)

action_std = None


#####################################################


## Note : print/log frequencies should be > than max_ep_len


################ PPO hyperparameters ################

update_timestep = max_ep_len * 3      # update policy every n timesteps
K_epochs = 30               # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
gamma = 0.99                # discount factor

lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network

random_seed = 0         # set random seed if required (0 = no random seed)

#####################################################
```





