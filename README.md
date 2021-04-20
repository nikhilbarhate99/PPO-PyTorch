# PPO-PyTorch

### UPDATE [April 2021] : 

- merged discrete and continuous algorithms
- added linear decaying for the continuous action space `action_std`; to make training more stable for complex environments
- added different learning rates for actor and critic
- episodes, timesteps and rewards are now logged in `.csv` files
- utils to plot graphs from log files
- utils to test and make gifs from preTrained networks
- `PPO_colab.ipynb` combining all the files to train / test / plot graphs / make gifs on google colab in a convenient jupyter-notebook

#### [Open `PPO_colab.ipynb` in Google Colab](https://colab.research.google.com/github/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_colab.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_colab.ipynb)


## Introduction

This repository provides a Minimal PyTorch implementation of Proximal Policy Optimization (PPO) with clipped objective for OpenAI gym environments. It is primarily intended for beginners in [Reinforcement Learning](https://en.wikipedia.org/wiki/Reinforcement_learning) for understanding the PPO algorithm. It can still be used for complex environments but may require some hyperparameter-tuning or changes in the code.

To keep the training procedure simple : 
  - It has a **constant standard deviation** for the output action distribution (**multivariate normal with diagonal covariance matrix**) for the continuous environments, i.e. it is a hyperparameter and NOT a trainable parameter. However, it is **linearly decayed**. (action_std significantly affects performance)
  - It uses simple **monte-carlo estimate** for calculating advantages and NOT Generalized Advantage Estimate (check out the OpenAI spinning up implementation for that).
  - It is a **single threaded implementation**, i.e. only one worker collects experience. [One of the older forks](https://github.com/rhklite/Parallel-PPO-PyTorch) of this repository has been modified to have Parallel workers

A concise explaination of PPO algorithm can be found [here](https://stackoverflow.com/questions/46422845/what-is-the-way-to-understand-proximal-policy-optimization-algorithm-in-rl)


## Usage

- To train a new network : run `train.py`
- To test a preTrained network : run `test.py`
- To plot graphs using log files : run `plot_graph.py`
- To save images for gif and make gif using a preTrained network : run `make_gif.py`
- All parameters and hyperparamters to control training / testing / graphs / gifs are in their respective `.py` file
- `PPO_colab.ipynb` combines all the files in a jupyter-notebook
- All the **hyperparameters used for training (preTrained) policies are listed** in the [`README.md` in PPO_preTrained directory](https://github.com/nikhilbarhate99/PPO-PyTorch/tree/master/PPO_preTrained)

#### Note :
  - if the environment runs on CPU, use CPU as device for faster training. Box-2d and Roboschool run on CPU and training them on GPU device will be significantly slower because the data will be moved between CPU and GPU often

## Citing 

Please use this bibtex if you want to cite this repository in your publications :

    @misc{pytorch_minimal_ppo,
        author = {Barhate, Nikhil},
        title = {Minimal PyTorch Implementation of Proximal Policy Optimization},
        year = {2021},
        publisher = {GitHub},
        journal = {GitHub repository},
        howpublished = {\url{https://github.com/nikhilbarhate99/PPO-PyTorch}},
    }

## Results

| PPO Continuous RoboschoolHalfCheetah-v1  | PPO Continuous RoboschoolHalfCheetah-v1 |
| :-------------------------:|:-------------------------: |
| ![](https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_gifs/RoboschoolHalfCheetah-v1/PPO_RoboschoolHalfCheetah-v1_gif_0.gif) |  ![](https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_figs/RoboschoolHalfCheetah-v1/PPO_RoboschoolHalfCheetah-v1_fig_0.png) |


| PPO Continuous RoboschoolHopper-v1  | PPO Continuous RoboschoolHopper-v1 |
| :-------------------------:|:-------------------------: |
| ![](https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_gifs/RoboschoolHopper-v1/PPO_RoboschoolHopper-v1_gif_0.gif) |  ![](https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_figs/RoboschoolHopper-v1/PPO_RoboschoolHopper-v1_fig_0.png) |


| PPO Continuous RoboschoolWalker2d-v1  | PPO Continuous RoboschoolWalker2d-v1 |
| :-------------------------:|:-------------------------: |
| ![](https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_gifs/RoboschoolWalker2d-v1/PPO_RoboschoolWalker2d-v1_gif_0.gif) |  ![](https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_figs/RoboschoolWalker2d-v1/PPO_RoboschoolWalker2d-v1_fig_0.png) |


| PPO Continuous BipedalWalker-v2  | PPO Continuous BipedalWalker-v2 |
| :-------------------------:|:-------------------------: |
| ![](https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_gifs/BipedalWalker-v2/PPO_BipedalWalker-v2_gif_0.gif) |  ![](https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_figs/BipedalWalker-v2/PPO_BipedalWalker-v2_fig_0.png) |


| PPO Discrete CartPole-v1  | PPO Discrete CartPole-v1 |
| :-------------------------:|:-------------------------: |
| ![](https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_gifs/CartPole-v1/PPO_CartPole-v1_gif_0.gif) |  ![](https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_figs/CartPole-v1/PPO_CartPole-v1_fig_0.png) |


| PPO Discrete LunarLander-v2  | PPO Discrete LunarLander-v2 |
| :-------------------------:|:-------------------------: |
| ![](https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_gifs/LunarLander-v2/PPO_LunarLander-v2_gif_0.gif) |  ![](https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_figs/LunarLander-v2/PPO_LunarLander-v2_fig_0.png) |


## Dependencies
Trained and Tested on:
```
Python 3
PyTorch
NumPy
gym
```
Training Environments 
```
Box-2d
Roboschool
pybullet
```
Graphs and gifs
```
pandas
matplotlib
Pillow
```


## References

- [PPO paper](https://arxiv.org/abs/1707.06347)
- [OpenAI Spinning up](https://spinningup.openai.com/en/latest/)


