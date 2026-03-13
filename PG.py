import os
import random
import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt

from keras import backend as K
import copy
from gymnasium.wrappers import RecordVideo

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")
tf.keras.utils.disable_interactive_logging()


class Agent:
    # PPO Main Optimization Algorithm
    def __init__(self, env_name):
        os.makedirs("models",exist_ok=True)
        self.env_name = env_name
        self.env = gym.make(env_name, max_episode_steps=1000)
        self.action_size = int(self.env.action_space.n)
        self.state_size = self.env.observation_space.shape
        self.EPISODES = 10000 # total episodes
        self.episode = 0 # current episode
        self.lr = 0.00025
        self.gamma = 0.99
        self.epochs = 3 # training epochs
        self.shuffle=True
        self.actor_optimizer = tf.keras.optimizers.Adam(self.lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(self.lr)

        self.actor = tf.keras.Sequential([
            Dense(128, activation='relu'),
            Dense(128, activation='relu'),
            Dense(self.action_size, activation='softmax')
        ])
        self.critic = tf.keras.Sequential([
            Dense(128, activation='relu'),
            Dense(128, activation='relu'),
            Dense(1)
        ])

    def run(self): 
        # self.load_model("models5/actor_ep5500.keras", "models5/critic_ep5500.keras")
        scores = []
        state,_ = self.env.reset()
        state = np.reshape(state, [1, self.state_size[0]])
        done=False
        score=0
        fig1 = plt.figure(1)
        fig2 = plt.figure(2)
        while True:
            # Instantiate or reset games memory
            states, next_states, rewards, actions, action_probs, dones = [], [], [], [], [], []
            done = False
            score=0
            steps=0
            with tf.GradientTape(persistent=True) as tape:
                while not done:
                    steps+=1
                    # Actor picks an action
                    action_prob = tf.squeeze(self.actor(state), axis=0)
                    action_prob_np = action_prob.numpy() / np.sum(action_prob.numpy())
                    action = np.random.choice(self.action_size, p=action_prob_np)

                    # Retrieve new state, reward, and whether the state is terminal
                    next_state, reward, terminated, truncated, _ = self.env.step(action)

                    done = terminated | truncated
                    # Memorize (state, action, reward) for training
                    states.append(state)
                    next_states.append(np.reshape(next_state, [1, self.state_size[0]]))
                    rewards.append(reward)
                    dones.append(done)
                    action_probs.append(action_prob[action])
                    # Update current state
                    state = np.reshape(next_state, [1, self.state_size[0]])
                    score += reward
                    if done:
                        self.episode += 1
                        # log #############
                        scores.append(score)
                        if(self.episode%20 == 0):
                            print("///////////////////////")
                            print("episode: {}/{}, score: {}".format(self.episode, self.EPISODES, score))
                            print("steps:", steps)

                            last = 100
                            window = 100
                            plt.figure(1)
                            plt.clf()
                            plt.plot(scores[-last:], label="Last 100 Episode Rewards")
                            plt.xlabel("Episode")
                            plt.ylabel("Reward")
                            plt.title("Recent Performance")
                            plt.legend()
                            plt.pause(0.1)


                            avg_scores = [
                                np.mean(scores[i:i+window])
                                for i in range(0, len(scores), window)
                            ]

                            plt.figure(2)
                            plt.clf()
                            plt.plot(avg_scores, label="Avg Reward per 100 Episodes")
                            plt.xlabel("Episode Block(100)")
                            plt.ylabel("Average Reward")
                            plt.title("Long-Term Progress")
                            plt.legend()
                            plt.pause(0.1)
                        #################
                        # update policy
                        rewards = np.array(rewards, dtype=np.float32)
                        states_array = np.vstack(states)
                        next_states_array = np.vstack(next_states)
                        for _ in range(self.epochs):
                            state_values = self.critic(states_array)
                            next_state_values = self.critic(next_states_array)
            
                            advantages = rewards + self.gamma * next_state_values - state_values
                            advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8) # batch normalize
                            
                            actor_loss = -tf.math.log(action_probs) * advantages
                            actor_loss = tf.reduce_mean(actor_loss) # mean over batch

                            critic_loss = tf.square(advantages)
                            critic_loss = tf.reduce_mean(critic_loss)

                            actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
                            critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
                            
                            self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
                            self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))


                        state,_ = self.env.reset()
                        state = np.reshape(state, [1, self.state_size[0]])
                        # Save every 250 episodes
                        if (self.episode) % 250 == 0:
                            self.actor.save(f"models5/actor_ep{self.episode}_comp.keras")
                            self.critic.save(f"models5/critic_ep{self.episode}_comp.keras")
                            print(f"Saved model at episode {self.episode}")
            if self.episode >= self.EPISODES:
                plt.figure(2)
                plt.savefig("running_score.png")
                break
        self.env.close()

    def test(self, test_episodes = 20):
        # self.load_model("models2/actor_avg_above_0.h5", "models2/critic_avg_above_0.h5")
        self.load_model("models5/actor_ep5500.keras", "models5/critic_ep5500.keras")
        test_env = RecordVideo(gym.make("LunarLander-v3", render_mode="rgb_array", max_episode_steps=1000), "video4_test", episode_trigger=lambda ep: True)
        for e in range(test_episodes):
            state,_ = test_env.reset()
            state = np.reshape(state, [1, self.state_size[0]])
            done = False
            score = 0
            steps=0
            while not done:
                steps+=1
                action_prob = tf.squeeze(self.actor(state, training=False), axis=0)
                action_prob_np = action_prob.numpy() / np.sum(action_prob.numpy())
                if steps==10:
                    print(action_prob_np)
                action = np.argmax(action_prob_np)
                state, reward, terminated, truncated, _ = test_env.step(action)
                done = terminated|truncated
                x_pos = state[0]
                y_pos = state[1]
                left_leg = state[6]
                right_leg = state[7]
                y_vel = state[3]
                x_vel = state[2]
                angle = state[4]
                ang_vel = state[5]

                
                state = np.reshape(state, [1, self.state_size[0]])
                score += reward
                if done:
                    print("x_pos: {}, y_pos:{}".format(x_pos,y_pos))
                    print("y_vel: ", y_vel, " angle: ", angle, " ang_vel: ", ang_vel)
                    print("episode: {}/{}, score: {}".format(e, test_episodes, score))
                    print(steps)
                    break
        
        test_env.close()

    def load_model(self, actor_path='models/actor_model.h5', critic_path='models/critic_model.h5'):
        if os.path.exists(actor_path):
            self.actor = load_model(actor_path, compile=False)
            print(f"Loaded Actor model from {actor_path}")
        else:
            print(f"Actor model file not found at {actor_path}")

        if os.path.exists(critic_path):
            self.critic = load_model(critic_path, compile=False)
            print(f"Loaded Critic model from {critic_path}")
        else:
            print(f"Critic model file not found at {critic_path}")

        # Recompile manually using custom loss methods from Actor_Model and Critic_Model
        self.actor.compile(
            optimizer=self.actor_optimizer
        )

        self.critic.compile(
            optimizer=self.critic_optimizer
        )
        print("Loaded and recompiled Actor and Critic models.")
            
if __name__ == "__main__":
    env_name = 'LunarLander-v3'
    agent = Agent(env_name)
    print("running....")
    agent.run() 
    # agent.test()