import gym
import numpy as np
import policynet

env = gym.make('CartPole-v0')

net = policynet.policynet(10, 2)


for iteration in range(500):

    observation = env.reset()

    done = False
    t = 0

    actions_i = []
    rewards_i = []
    observations_i = []

    while(not done):

        t = t + 1
        if iteration % 10 == 0:
            env.render()


        #action = env.action_space.sample()
        action = net.best_action(observation)

      #  print(action)

        observation, reward, done, info = env.step(action)

        actions_i = actions_i + [action]
        observations_i = observations_i + [observation]
        rewards_i = rewards_i + [reward]

    loss = net.update_policy(observations_i, actions_i, rewards_i)

    if iteration % 10 == 0:
        r = sum(rewards_i)
        print ("Reward: {}".format(r))
        print ("Loss: {}".format(loss))