# PPO-Panda

### UPDATE [April 2022] : 

- Modified [this commit](https://github.com/nikhilbarhate99/PPO-PyTorch) created by Nikhil Barhate to accomodate the Franka Emika Panda Robot.
- Installed Panda-Gym
- Adapted the classes to account for the difference in return after the step  

#### [Open `PPO_colab.ipynb` in Google Colab](https://colab.research.google.com/github/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_colab.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_colab.ipynb) to see original PPO implementation for roboschool


## Introduction

- See [this link](https://github.com/nikhilbarhate99/PPO-PyTorch) to review the details (learning rates, episode logging, utils, etc) of this implementation of PPO. 
- New modifications include to the reward figure, the rendering, etc
- No changes in PPO

## Usage

- To train a new network : run `train.py`
- To test a preTrained network : run `test.py`
- To plot graphs using log files : run `plot_graph.py`
- To save images for gif and make gif using a preTrained network : run `make_gif.py`
- All parameters and hyperparamters to control training / testing / graphs / gifs are in their respective `.py` file
- `PPO_colab.ipynb` combines all the files in a jupyter-notebook
- All the **hyperparameters used for training (preTrained) policies are listed** in the [`README.md` in PPO_preTrained directory](https://github.com/nikhilbarhate99/PPO-PyTorch/tree/master/PPO_preTrained)

#### Note :
  - if the environment runs on CPU, use CPU as device for faster training. 
  
## Citing 

Please use this bibtex if you want to cite this repository in your publications :

    @misc{ppo_panda,
        author = {Lobbezoo, Andrew},
        title = {PyTorch Implementation of Proximal Policy Optimization for the OpenAI Panda},
        year = {2022},
        publisher = {GitHub},
        journal = {GitHub repository},
        howpublished = {\url{https://github.com/alobbezoo/PPO-Panda}},
    }

## Results

| PPO Continuous PandaReachDense-v2  | PPO Continuous PandaReachDense-v2 |
| :-------------------------:|:-------------------------: |
| ![](https://github.com/alobbezoo/PPO-Panda/blob/b470118413237fd8d52055cb9880a3b5dfa17040/PPO_gifs/PandaReachDense-v2/PPO_PandaReachDense-v2_gif_0.gif) |  ![](https://github.com/alobbezoo/PPO-Panda/blob/master/PPO_figs/PandaReachDense-v2/PPO_PandaReachDense-v2_fig_0.png) |


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
gym
```
Graphs and gifs
```
pandas
matplotlib
Pillow
pyvirtualdisplay
python-opengl

```


## References

- [PPO paper](https://arxiv.org/abs/1707.06347)
- [OpenAI Spinning up](https://spinningup.openai.com/en/latest/)
- [PPO-PyTorch](https://github.com/nikhilbarhate99/PPO-PyTorch)


