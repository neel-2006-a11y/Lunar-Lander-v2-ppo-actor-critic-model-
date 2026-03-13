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
        self.state_size = self.env.observation_space.shape[0]

        self.EPISODES = 5000 # total episodes
        self.episode = 0 # current episode

        self.Clr = 3e-4
        self.Alr = 1e-4
        self.gamma = 0.99
        self.clip_epsilon = 0.2
        self.lamb = 0.95
        self.epochs = 5 # training epochs
        self.entropy_f = 0.1

        self.actor_optimizer = tf.keras.optimizers.Adam(self.Alr)
        self.critic_optimizer = tf.keras.optimizers.Adam(self.Clr)

        self.actor = tf.keras.Sequential([
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(self.action_size, activation='softmax')
        ])
        self.critic = tf.keras.Sequential([
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(1)
        ])

    def run(self): 
        # self.load_model("models3/actor_ep3000_ppo.keras", "models3/critic_ep3000_ppo.keras")
        scores = []
        state,_ = self.env.reset()
        state = np.reshape(state, [1, self.state_size])
        done=False
        score=0
        fig1 = plt.figure(1)
        fig2 = plt.figure(2)
        while True:
            # Instantiate or reset games memory
            states, next_states, rewards, actions, action_probs_selected, dones = [], [], [], [], [], []
            done = False
            score=0
            steps=0
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
                next_states.append(np.reshape(next_state, [1, self.state_size]))
                rewards.append(reward)
                actions.append(action)
                dones.append(done)
                action_probs_selected.append(action_prob_np[action])
                # Update current state
                state = np.reshape(next_state, [1, self.state_size])
                score += reward
                if done:
                    self.episode += 1
                    # log #############
                    scores.append(score)

                    last = 100
                    window = 100
                    terminal_window = 20
                    if(self.episode%terminal_window == 0):
                        print("///////////////////////")
                        print("episode: {}/{}, score: {}".format(self.episode, self.EPISODES, score))
                        print("steps:", steps)

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
                    rewards_np = np.array(rewards, dtype=np.float32)
                    dones_np = np.array(dones, dtype=np.float32)

                    # calculate values from critic
                    states_array = np.vstack(states)
                    next_states_array = np.vstack(next_states)

                    state_values = self.critic(states_array)
                    next_state_values = self.critic(next_states_array)

                    state_values = tf.squeeze(state_values)
                    next_state_values = tf.squeeze(next_state_values)

                    deltas = rewards_np + self.gamma * next_state_values * (np.ones_like(dones_np) - dones_np) - state_values
                    advantages = []
                    gae = 0.0

                    for delta, done_t in zip(reversed(deltas), reversed(dones_np)):
                        gae = delta + self.gamma * self.lamb * gae * (1-done_t)
                        advantages.insert(0,gae)

                    advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
                    # # batch norm 
                    # advantages = (advantages - tf.reduce_mean(advantages))
                    returns = advantages + state_values # critic should fit this

                    # convert np to tf
                    actions = tf.convert_to_tensor(actions,dtype=tf.int32)
                    old_action_probs_selected = tf.convert_to_tensor(action_probs_selected, dtype=tf.float32)

                    for e in range(self.epochs):
                        with tf.GradientTape(persistent=True) as tape:
                            
                            new_action_probs = self.actor(states_array)
                            # print(new_action_probs.shape)

                            indices = tf.stack([tf.range(tf.shape(actions)[0]), actions],axis=1)
                            new_action_probs_selected = tf.gather_nd(new_action_probs, indices)
                            new_log_probs = tf.math.log(new_action_probs_selected + 1e-8)
                            old_log_probs = tf.math.log(old_action_probs_selected + 1e-8)

                            ratio = tf.exp(new_log_probs - old_log_probs)
                            clipped_ratio = tf.clip_by_value(
                                ratio,
                                1 - self.clip_epsilon,
                                1 + self.clip_epsilon
                            )
                            surrogate1 = ratio * advantages
                            surrogate2 = clipped_ratio * advantages

                            actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

                            entropy = -tf.reduce_mean(
                                tf.reduce_sum(new_action_probs * tf.math.log(new_action_probs+1e-8),axis=1)
                            )
                            actor_loss = actor_loss - self.entropy_f * entropy

                            values = tf.squeeze(self.critic(states_array))
                            critic_loss = tf.reduce_mean(tf.square(returns - values))

                            if(self.episode%terminal_window == 0 and e==0):
                                print("Entropy Loss: ", entropy * self.entropy_f)
                                print("Total Actor Loss: ", actor_loss)
                                print("Critic Loss: ", critic_loss)

                            actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
                            critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
                            
                            self.actor_optimizer.apply_gradients(
                                zip(actor_gradients, self.actor.trainable_variables)
                            )
                            self.critic_optimizer.apply_gradients(
                                zip(critic_gradients, self.critic.trainable_variables)
                            )


                    state,_ = self.env.reset()
                    state = np.reshape(state, [1, self.state_size])
                    # Save every 250 episodes
                    if (self.episode) % 250 == 0:
                        self.actor.save(f"models4/actor_ep{self.episode}_ppo.keras")
                        self.critic.save(f"models4/critic_ep{self.episode}_ppo.keras")
                        print(f"Saved model at episode {self.episode}")
            if self.episode >= self.EPISODES:
                plt.figure(2)
                plt.savefig("running_score_ppo.png")
                break
        self.env.close()

    def test(self, test_episodes = 20):
        # self.load_model("models2/actor_avg_above_0.h5", "models2/critic_avg_above_0.h5")
        self.load_model("models4/actor_ep5000_ppo.keras", "models4/critic_ep5000_ppo.keras")
        test_env = RecordVideo(gym.make("LunarLander-v3", render_mode="rgb_array", max_episode_steps=1000), "video4_test", episode_trigger=lambda ep: True)
        for e in range(test_episodes):
            state,_ = test_env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            score = 0
            steps=0
            while not done:
                steps+=1
                action_prob = tf.squeeze(self.actor(state, training=False), axis=0)
                action_prob_np = action_prob.numpy() / np.sum(action_prob.numpy())
                if steps==10:
                    print(action_prob_np)
                # action = np.argmax(action_prob_np)
                action = np.random.choice(self.action_size, p=action_prob_np)
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

                
                state = np.reshape(state, [1, self.state_size])
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
    # agent.run() 
    agent.test()