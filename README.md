# PPO-PyTorch
Simple and beginner friendly PyTorch implementation of [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) with clipped objective for OpenAI gym environment.

## Usage
- To test a preTrained network : run `test.py`
- If you are trying to train it on a environment where action dimension = 1, make sure to check the tensor dimensions while updating in the update function, since I have used `torch.squeeze()` quite a few times. `torch.squeeze()` squeezes the tensor such that there are no dimensions of length = 1.([more info](https://pytorch.org/docs/stable/torch.html?highlight=torch%20squeeze#torch.squeeze))

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

PPO Discrete LunarLander-v2 (1200 episodes)

![](https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/gif/PPO_LunarLander-v2.gif) 
