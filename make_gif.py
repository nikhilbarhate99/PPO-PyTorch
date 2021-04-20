import os
import glob
import time
from datetime import datetime

import torch
import numpy as np
from PIL import Image

import gym
import roboschool

# import pybullet_envs

from PPO import PPO




"""
One frame corresponding to each timestep is saved in a folder :

PPO_gif_images/env_name/000001.jpg
PPO_gif_images/env_name/000002.jpg
PPO_gif_images/env_name/000003.jpg
...
...
...


if this section is run multiple times or for multiple episodes for the same env_name;
then the saved images will be overwritten.

"""

############################# save images for gif ##############################


def save_gif_images(env_name, has_continuous_action_space, max_ep_len, action_std):

	print("============================================================================================")

	total_test_episodes = 1     # save gif for only one episode


	K_epochs = 80               # update policy for K epochs
	eps_clip = 0.2              # clip parameter for PPO
	gamma = 0.99                # discount factor

	lr_actor = 0.0003         # learning rate for actor
	lr_critic = 0.001         # learning rate for critic


	env = gym.make(env_name)

	# state space dimension
	state_dim = env.observation_space.shape[0]

	# action space dimension
	if has_continuous_action_space:
		action_dim = env.action_space.shape[0]
	else:
		action_dim = env.action_space.n



	# make directory for saving gif images
	gif_images_dir = "PPO_gif_images" + '/'
	if not os.path.exists(gif_images_dir):
		os.makedirs(gif_images_dir)

	# make environment directory for saving gif images
	gif_images_dir = gif_images_dir + '/' + env_name + '/'
	if not os.path.exists(gif_images_dir):
		os.makedirs(gif_images_dir)

	# make directory for gif
	gif_dir = "PPO_gifs" + '/'
	if not os.path.exists(gif_dir):
		os.makedirs(gif_dir)

	# make environment directory for gif
	gif_dir = gif_dir + '/' + env_name  + '/'
	if not os.path.exists(gif_dir):
		os.makedirs(gif_dir)



	ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)


	# preTrained weights directory

	random_seed = 0             #### set this to load a particular checkpoint trained on random seed
	run_num_pretrained = 0      #### set this to load a particular checkpoint num


	directory = "PPO_preTrained" + '/' + env_name + '/'
	checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
	print("loading network from : " + checkpoint_path)

	ppo_agent.load(checkpoint_path)

	print("--------------------------------------------------------------------------------------------")



	test_running_reward = 0

	for ep in range(1, total_test_episodes+1):

		ep_reward = 0
		state = env.reset()

		for t in range(1, max_ep_len+1):
		    action = ppo_agent.select_action(state)
		    state, reward, done, _ = env.step(action)
		    ep_reward += reward

		    img = env.render(mode = 'rgb_array')

		    img = Image.fromarray(img)
		    img.save(gif_images_dir + '/' + str(t).zfill(6) + '.jpg')

		    if done:
		        break

		# clear buffer
		ppo_agent.buffer.clear()

		test_running_reward +=  ep_reward
		print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
		ep_reward = 0



	env.close()



	print("============================================================================================")

	print("total number of frames / timesteps / images saved : ", t)

	avg_test_reward = test_running_reward / total_test_episodes
	avg_test_reward = round(avg_test_reward, 2)
	print("average test reward : " + str(avg_test_reward))

	print("============================================================================================")







######################## generate gif from saved images ########################

def save_gif(env_name):

	print("============================================================================================")

	gif_num = 0     #### change this to prevent overwriting gifs in same env_name folder

	# adjust following parameters to get desired duration, size (bytes) and smoothness of gif
	total_timesteps = 300
	step = 10
	frame_duration = 150


	# input images
	gif_images_dir = "PPO_gif_images/" + env_name + '/*.jpg'


	# ouput gif path
	gif_dir = "PPO_gifs"
	if not os.path.exists(gif_dir):
		os.makedirs(gif_dir)

	gif_dir = gif_dir + '/' + env_name
	if not os.path.exists(gif_dir):
		os.makedirs(gif_dir)

	gif_path = gif_dir + '/PPO_' + env_name + '_gif_' + str(gif_num) + '.gif'



	img_paths = sorted(glob.glob(gif_images_dir))
	img_paths = img_paths[:total_timesteps]
	img_paths = img_paths[::step]


	print("total frames in gif : ", len(img_paths))
	print("total duration of gif : " + str(round(len(img_paths) * frame_duration / 1000, 2)) + " seconds")



	# save gif
	img, *imgs = [Image.open(f) for f in img_paths]
	img.save(fp=gif_path, format='GIF', append_images=imgs, save_all=True, optimize=True, duration=frame_duration, loop=0)

	print("saved gif at : ", gif_path)



	print("============================================================================================")






############################# check gif byte size ##############################

def list_gif_size(env_name):

	print("============================================================================================")

	gif_dir = "PPO_gifs/" + env_name + '/*.gif'

	gif_paths = sorted(glob.glob(gif_dir))

	for gif_path in gif_paths:
		file_size = os.path.getsize(gif_path)
		print(gif_path + '\t\t' + str(round(file_size / (1024 * 1024), 2)) + " MB")


	print("============================================================================================")





if __name__ == '__main__':


	# env_name = "CartPole-v1"
	# has_continuous_action_space = False
	# max_ep_len = 400
	# action_std = None


	# env_name = "LunarLander-v2"
	# has_continuous_action_space = False
	# max_ep_len = 500
	# action_std = None


	# env_name = "BipedalWalker-v2"
	# has_continuous_action_space = True
	# max_ep_len = 1500           # max timesteps in one episode
	# action_std = 0.1            # set same std for action distribution which was used while saving


	# env_name = "RoboschoolWalker2d-v1"
	# has_continuous_action_space = True
	# max_ep_len = 1000           # max timesteps in one episode
	# action_std = 0.1            # set same std for action distribution which was used while saving


	env_name = "RoboschoolHalfCheetah-v1"
	has_continuous_action_space = True
	max_ep_len = 1000           # max timesteps in one episode
	action_std = 0.1            # set same std for action distribution which was used while saving


	# env_name = "RoboschoolHopper-v1"
	# has_continuous_action_space = True
	# max_ep_len = 1000           # max timesteps in one episode
	# action_std = 0.1            # set same std for action distribution which was used while saving

	# save .jpg images in PPO_gif_images folder
	save_gif_images(env_name, has_continuous_action_space, max_ep_len, action_std)

	# save .gif in PPO_gifs folder using .jpg images
	save_gif(env_name)

	# list byte size (in MB) of gifs in one "PPO_gif/env_name/" folder
	list_gif_size(env_name)
