# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)


# TensorFlow ≥2.0-preview is required
import tensorflow as tf
# from tensorflow import keras
# assert tf.__version__ >= "2.0"

# Common imports
import numpy as np
import os

import gym
from gym import spaces
import pandas as pd
import random

import matplotlib.pyplot as plt

from myStockEnv import StockTradingEnv

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.ddpg.policies import MlpPolicy as ddpg_MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
from stable_baselines import A2C
from stable_baselines import DDPG
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec

from stable_baselines import SAC
from stable_baselines.sac.policies import MlpPolicy as SAC_MlpPolicy

MAX_NET_WORTH = 2147483647

b=np.array([3, 1])
def random_policy(obs):
    action =  np.atleast_2d(np.random.random_sample()*b)
    return action

if __name__ == '__main__':

	## Train
	df_stock_prices=pd.read_csv('BABA_train.csv')
	# print(df_stock_prices.shape[0])
	# df_stock_prices=pd.read_csv('AAPL.csv')
	df_stock_prices['Date']=pd.to_datetime(df_stock_prices['Date'])
	df_stock_prices=df_stock_prices.sort_values(by='Date')

	env = StockTradingEnv(df_stock_prices)
	INITIAL_ACCOUNT_BALANCE = env.INITIAL_ACCOUNT_BALANCE
	env = DummyVecEnv([lambda: env])

	# the noise objects for DDPG
	# n_actions = env.action_space.shape[-1]
	# param_noise = None
	# action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(2.5) * np.ones(n_actions))
	# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=float(3) * np.ones(n_actions))

	# model = PPO2(MlpPolicy, env, gamma=0.998, n_steps=16, verbose=0)
	# model = DDPG(ddpg_MlpPolicy, env, gamma=0.995, verbose=1, param_noise=param_noise, action_noise=action_noise)
	model = A2C(MlpPolicy, env, gamma=0.99, n_steps=16, verbose=0)
	# model.learn(total_timesteps=50000)
	# model = SAC(SAC_MlpPolicy, env, gamma=0.995, verbose=0)


	obs = env.reset()
	# model.learn(total_timesteps=100000)

	done = False
	list_train_net_earning_per_day = []
	list_train_return = []
	r = 0
	policy_ddpg = True
	for i in range(1000):
		if(i%100==0):
			print('iteration', i)
			model.learn(total_timesteps=5000)
		else:
			obs = env.reset()
			done = False
			while(not done):
				action, _states = model.predict(obs)
				# print(action)
				# action = random_policy(obs)
				obs, rewards, dones, info = env.step(action)
				done = dones[0]
				r += rewards[0]
				if(done): 
					net_worth_at_end = env.get_attr('net_worth_done')[0]
					# print('net_worth_at_end', net_worth_at_end)
					counter_at_end = env.get_attr('counter_done')[0]
					avg_earning_per_day = (net_worth_at_end-INITIAL_ACCOUNT_BALANCE)/counter_at_end
					# print((net_worth_at_end-INITIAL_ACCOUNT_BALANCE)/counter_at_end*1000)
					list_train_net_earning_per_day.append(avg_earning_per_day)
					list_train_return.append(r)
					r = 0
					break


	train_array_net_earning_per_day = np.array(list_train_net_earning_per_day)
	train_array_return = np.array(list_train_return)
	np.save('train_A2C_array_net_earning_per_day', train_array_net_earning_per_day)
	plt.figure(0)
	plt.plot(list_train_net_earning_per_day)


	
	## Test
	print('Test begins')
	df_stock_prices=pd.read_csv('BABA_test.csv')
	# df_stock_prices=pd.read_csv('AAPL.csv')
	df_stock_prices['Date']=pd.to_datetime(df_stock_prices['Date'])
	df_stock_prices=df_stock_prices.sort_values(by='Date')

	env = StockTradingEnv(df_stock_prices)
	INITIAL_ACCOUNT_BALANCE = env.INITIAL_ACCOUNT_BALANCE
	env = DummyVecEnv([lambda: env])
	obs = env.reset()

	# done = False
	list_test_net_earning_per_day = []
	list_test_return = []
	r = 0
	for i in range(100):
		obs = env.reset()
		done = False
		while(not done):
			action, _states = model.predict(obs)
			# action = random_policy(obs)
			obs, rewards, dones, info = env.step(action)
			done = dones[0]
			r += rewards[0]
			if(done): 
				net_worth_at_end = env.get_attr('net_worth_done')[0]
				counter_at_end = env.get_attr('counter_done')[0]
				avg_earning_per_day = (net_worth_at_end-INITIAL_ACCOUNT_BALANCE)/counter_at_end
				list_test_net_earning_per_day.append(avg_earning_per_day)
				list_test_return.append(r)
				r = 0
				break


	array_net_earning_per_day = np.array(list_test_net_earning_per_day)
	np.save('test_A2C_array_net_earning_per_day', array_net_earning_per_day)
	plt.figure(1)
	plt.plot(list_test_net_earning_per_day)
	plt.show()