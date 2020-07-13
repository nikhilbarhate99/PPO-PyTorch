# PPO-PyTorch
Minimal PyTorch implementation of Proximal Policy Optimization with clipped objective for OpenAI gym environments.

## Usage

- To test a preTrained network : run `test.py` or `test_continuous.py`
- To train a new network : run `PPO.py` or `PPO_continuous.py`
- All the hyperparameters are in the `PPO.py` or `PPO_continuous.py` file
- If you are trying to train it on a environment where action dimension = 1, make sure to check the tensor dimensions in the update function of PPO class, since I have used `torch.squeeze()` quite a few times. `torch.squeeze()` squeezes the tensor such that there are no dimensions of length = 1 ([more info](https://pytorch.org/docs/stable/torch.html?highlight=torch%20squeeze#torch.squeeze)).
- Number of actors for collecting experience = 1. This could be changed by creating multiple instances of ActorCritic networks in the PPO class and using them to collect experience (like A3C and standard PPO).

## Dependencies
Trained and tested on:
```
Python 3.6
PyTorch 1.0
NumPy 1.15.3
gym 0.10.8
Pillow 5.3.0
```

## Results

PPO Discrete LunarLander-v2 (1200 episodes)           |  PPO Continuous BipedalWalker-v2 (4000 episodes)
:-------------------------:|:-------------------------:
![](https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/gif/PPO_LunarLander-v2.gif) |  ![](https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/gif/PPO_BipedalWalker-v2.gif)


## References

- PPO [paper](https://arxiv.org/abs/1707.06347)
- [OpenAI Spinning up](https://spinningup.openai.com/en/latest/)
