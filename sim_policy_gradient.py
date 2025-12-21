import os
import random
import gymnasium as gym
import numpy as np
import tensorflow as tf
# tf.compat.v1.disable_eager_execution() # usually using this for fastest performance
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import backend as K
import copy
from gymnasium.wrappers import RecordVideo
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")
tf.keras.utils.disable_interactive_logging()

class Actor_Model:
    def __init__(self, input_shape, action_space, learning_rate, optimizer):
        X_input = Input(input_shape)
        self.action_space = action_space
        self.init_entropy_loss = 0.005
        self.decay_rate = 0.995
        self.ENTROPY_LOSS = self.init_entropy_loss
        self.min_value = 0.0001

        X = Dense(512, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_input)
        X = Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        X = Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        output = Dense(self.action_space, activation="softmax")(X)

        self.Actor = Model(inputs = X_input, outputs = output)
        self.Actor.compile(loss=self.ppo_loss, optimizer=optimizer(learning_rate=learning_rate))
    
    def ppo_loss(self, y_true, y_pred):
        # Defined in https://arxiv.org/abs/1707.06347
        # y_true=[advantages, predictions, actions]
        advantages, prediction_picks, actions = y_true[:, :1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space:]
        LOSS_CLIPPING = 0.2
        
        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = K.clip(prob, 1e-10, 1.0)
        old_prob = K.clip(old_prob, 1e-10, 1.0)

        ratio = K.exp(K.log(prob) - K.log(old_prob))
        
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages

        actor_loss = -K.mean(K.minimum(p1, p2))

        entropy = -(y_pred * K.log(y_pred + 1e-10))
        entropy = self.ENTROPY_LOSS * K.mean(entropy)
        
        total_loss = actor_loss-entropy

        return total_loss
    def predict(self, state):
        return self.Actor.predict(state)
    
class Critic_Model:
    def __init__(self, input_shape, action_space, learning_rate, optimizer):
        X_input = Input(input_shape)

        V = Dense(512, activation="relu", kernel_initializer='he_uniform')(X_input)
        V = Dense(256, activation="relu", kernel_initializer='he_uniform')(V)
        V = Dense(64, activation="relu", kernel_initializer='he_uniform')(V)
        value = Dense(1, activation=None)(V)

        self.Critic = Model(inputs=X_input, outputs = value)
        self.Critic.compile(loss=self.critic_PPO2_loss, optimizer=optimizer(learning_rate=learning_rate))

    def critic_PPO2_loss(self, y_true, y_pred):
        # y_true =[target,values]
        LOSS_CLIPPING = 5

        # Suppose y_true is shape (batch_size, 2)
        target = y_true[:, 0]
        old_values = y_true[:, 1]
        returns = target-old_values
        
        clipped_value_loss = old_values + tf.clip_by_value(y_pred - old_values, -LOSS_CLIPPING, LOSS_CLIPPING)
        v_loss1 = tf.square(returns - clipped_value_loss)
        v_loss2 = tf.square(returns - y_pred)
        value_loss = 0.5 * tf.reduce_mean(tf.maximum(v_loss1, v_loss2))
        #value_loss = K.mean((y_true - y_pred) ** 2) # standard PPO loss
        return value_loss

    
    def predict(self, state):
        return self.Critic.predict(state)

class PPOAgent:
    # PPO Main Optimization Algorithm
    def __init__(self, env_name):
        os.makedirs("models_raw",exist_ok=True)
        # Initialization
        # Environment and PPO parameters
        self.env_name = env_name       
        self.env = gym.make(env_name, max_episode_steps=600)
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.shape
        self.EPISODES = 15000 # total episodes to train through all environments
        self.episode = 0 # used to track the episodes total count of episodes played through all thread environments
        self.lr = 0.00025
        self.epochs = 3 # training epochs
        self.shuffle=True
        self.optimizer = Adam

        # Create Actor-Critic network models
        self.Actor = Actor_Model(input_shape=self.state_size, action_space = self.action_size, learning_rate=self.lr, optimizer = self.optimizer)
        self.Critic = Critic_Model(input_shape=self.state_size, action_space = self.action_size, learning_rate=self.lr, optimizer = self.optimizer)
    
    def act(self, state):
        """ example:
        pred = np.array([0.05, 0.85, 0.1])
        action_size = 3
        np.random.choice(a, p=pred)
        result>>> 1, because it has the highest probability to be taken
        """
        # Use the network to predict the next action to take, using the model
        prediction = self.Actor.predict(state)[0]
        action = np.random.choice(self.action_size, p=prediction)
        action_onehot = np.zeros([self.action_size])
        action_onehot[action] = 1
        return action, action_onehot, prediction
    
    def discount_rewards(self, reward):#gaes is better
        # Compute the gamma-discounted rewards over an episode
        # We apply the discount and normalize it to avoid big variability of rewards
        gamma = 0.99    # discount rate
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0,len(reward))):
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r) # normalizing the result
        discounted_r /= (np.std(discounted_r) + 1e-8) # divide by standard deviation
        return discounted_r
    
    def get_gaes(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.9, normalize=True):
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values

        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target) # [advantages,returns]
    
    def replay(self, states, actions, rewards, predictions, dones, next_states):
        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        # Get Critic network predictions 
        values = self.Critic.predict(states)
        next_values = self.Critic.predict(next_states)

        # Compute discounted rewards and advantages
        advantages, returns = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values), normalize=False)
        returns = np.hstack([returns,values])

        # stack everything to numpy array
        # pack all advantages, predictions and actions to y_true and when they are received
        y_true = np.hstack([advantages, predictions, actions])
        
        # training Actor and Critic networks
        a_loss = self.Actor.Actor.fit(states, y_true, epochs=self.epochs, verbose=0, shuffle=self.shuffle)
        c_loss = self.Critic.Critic.fit(states, returns, epochs=self.epochs, verbose=0, shuffle=self.shuffle)

    def run(self): # train only when episode is finished
        # self.load_model("models_raw/actor_ep5200.h5", "models_raw/critic_ep5200.h5")
        scores = []
        state,_ = self.env.reset()
        prev_y = state[1]
        state = np.reshape(state, [1, self.state_size[0]])
        done=False
        score=0
        while True:
            # Instantiate or reset games memory
            states, next_states, actions, rewards, predictions, dones = [], [], [], [], [], []
            done = False
            score=0
            steps = 0
            while not done:
                steps+=1
                # Actor picks an action
                action, action_onehot, prediction = self.act(state)
                # Retrieve new state, reward, and whether the state is terminal
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                # Reward shaping
                x_pos = state[0][0]
                y_pos = state[0][1]
                left_leg = state[0][6]
                right_leg = state[0][7]
                y_vel = state[0][3]
                x_vel = state[0][2]
                angle = state[0][4]
                ang_vel = state[0][5]

                #centering
                reward-=abs(x_pos)*0.2
                
                # altitude shaping
                reward-=0.2*y_pos
                reward-=0.2*y_pos

                # velocity shaping
                if y_pos>0.3:
                    reward-=0.1*abs(y_vel)


                # angle shaping
                reward-=0.1*abs(angle)
                
                done = terminated | truncated
                # Memorize (state, action, reward) for training
                states.append(state)
                next_states.append(np.reshape(next_state, [1, self.state_size[0]]))
                actions.append(action_onehot)
                rewards.append(reward)
                dones.append(done)
                predictions.append(prediction)
                # Update current state
                state = np.reshape(next_state, [1, self.state_size[0]])
                score += reward
                if done:
                    print("///////////////////////")
                    print("x_pos: ",x_pos, "y_pos: ", y_pos, "angle: ", angle, "ang_vel: ", ang_vel)
                    print("y_vel",y_vel)
                    self.episode += 1
                    print("episode: {}/{}, score: {}".format(self.episode, self.EPISODES, score))
                    print("steps:", steps)
                    
                    self.replay(states, actions, rewards, predictions, dones, next_states)
                    state,_ = self.env.reset()
                    prev_y = state[1]
                    state = np.reshape(state, [1, self.state_size[0]])
                    scores.append(score)
                    # Save every 200 episodes
                    if (self.episode) % 200 == 0:
                        self.Actor.Actor.save(f"models_pg/actor_ep{self.episode}.h5")
                        self.Critic.Critic.save(f"models_pg/critic_ep{self.episode}.h5")
                        print(f"✅ Saved model at episode {self.episode}")
                        # Save if average score of last 50 episodes > 0
                    # if len(scores) >= 50:
                    #     avg_score = np.mean(scores[-50:])
                    #     if avg_score > 0:
                    #         self.Actor.Actor.save("models2/actor_avg_above_0.h5")
                    #         self.Critic.Critic.save("models2/critic_avg_above_0.h5")
                    #         print(f"✅ Saved model — avg score > 0: {avg_score:.2f}")

            if self.episode >= self.EPISODES:
                break
        self.env.close()

    def test(self, test_episodes = 20):
        self.load_model("models_raw/actor_ep4800.h5", "models_raw/critic_ep4800.h5")
        test_env = RecordVideo(gym.make("LunarLander-v3", render_mode="rgb_array", max_episode_steps=1000), "video4_test", episode_trigger=lambda ep: True)
        for e in range(test_episodes):
            state,_ = test_env.reset()
            state = np.reshape(state, [1, self.state_size[0]])
            done = False
            score = 0
            steps=0
            while not done:
                steps+=1
                prediction = self.Actor.predict(state)[0]
                action = np.random.choice(self.action_size, p=prediction)
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
            self.Actor.Actor = load_model(actor_path, compile=False)
            print(f"✅ Loaded Actor model from {actor_path}")
        else:
            print(f"⚠️ Actor model file not found at {actor_path}")

        if os.path.exists(critic_path):
            self.Critic.Critic = load_model(critic_path, compile=False)
            print(f"✅ Loaded Critic model from {critic_path}")
        else:
            print(f"⚠️ Critic model file not found at {critic_path}")

        # Recompile manually using custom loss methods from Actor_Model and Critic_Model
        self.Actor.Actor.compile(
            loss=self.Actor.ppo_loss,
            optimizer=Adam(learning_rate=self.lr)
        )

        self.Critic.Critic.compile(
            loss=self.Critic.critic_PPO2_loss,
            optimizer=Adam(learning_rate=self.lr)
        )
        print("✅ Loaded and recompiled Actor and Critic models.")
            
if __name__ == "__main__":
    env_name = 'LunarLander-v3'
    agent = PPOAgent(env_name)
    agent.run() # train as PPO, train every episode
    # agent.test()