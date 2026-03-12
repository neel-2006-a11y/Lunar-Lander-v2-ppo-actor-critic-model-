# Reinforcement Learning Lunar Lander Agent

This project implements a deep reinforcement learning agent trained to solve the LunarLander-v3 environment from the Gymnasium reinforcement learning library using TensorFlow and Keras. The system is based on an Actor–Critic architecture in which two neural networks are trained simultaneously: one network learns the policy that selects actions, while the other estimates the value of states in the environment.

The agent interacts with the simulation environment and improves its behavior through reward-driven learning. Over time it learns how to control the spacecraft thrusters in order to land safely between the landing flags while maintaining stable orientation and velocity.

---

# Features

## Actor–Critic Reinforcement Learning

The learning system uses two neural networks that work together during training.

The actor network outputs a probability distribution over possible actions in the environment. An action is sampled from this distribution, allowing the agent to explore different behaviors while learning.

The critic network estimates the value of the current state. This value is used to compute training advantages that guide the learning updates for the actor.

Key characteristics of the training system include policy-gradient optimization, advantage-based updates, simultaneous actor and critic learning, and episodic training with reward accumulation.

---

## Environment Interaction

The agent interacts with the Lunar Lander simulation environment provided by Gymnasium.

The environment provides an eight-dimensional observation vector that describes the current state of the lander. These values include horizontal and vertical position, horizontal and vertical velocity, the lander's angle and angular velocity, and two binary indicators representing contact between the lander's legs and the ground.

The action space consists of four discrete thruster controls: doing nothing, firing the left engine, firing the main engine, and firing the right engine. By learning when to activate these thrusters, the agent gradually develops a landing strategy.

---

## Training Pipeline

The training loop follows a structured process in which the agent repeatedly interacts with the environment and updates its neural networks.

At the start of each episode the environment is reset. The agent then observes the current state and selects actions using the actor network. As the episode progresses, states, actions, rewards, and next states are stored in memory.

After the episode ends, the critic network estimates the values of the collected states. These values are used to compute advantages that measure how much better or worse the observed rewards were compared to expected outcomes. The actor and critic networks are then updated using gradient descent.

This process repeats for thousands of episodes, gradually improving the policy learned by the actor network.

---

# Neural Network Architecture

The actor network receives the eight-dimensional state vector as input and passes it through two fully connected layers with ReLU activation. The final layer uses a softmax activation to produce a probability distribution over the four possible actions.

The critic network also receives the same state vector as input. It consists of a fully connected layer followed by a linear output layer that produces a single scalar value representing the estimated value of the state.

---

# Technologies

This project was implemented using Python and several machine learning and scientific computing libraries. The neural networks are built using TensorFlow and Keras, while the reinforcement learning environment is provided by Gymnasium. Numerical computation and array operations are handled using NumPy, and training performance is visualized using Matplotlib.

---

# Project Structure

The repository is organized into a small number of directories and files that separate the learning implementation from saved models and training outputs.

The models directory contains saved actor and critic checkpoints produced during training. A video directory stores recorded evaluation episodes. The main reinforcement learning implementation is contained in the agent script, and a reward curve image is produced during training to visualize learning progress.

---

# Example Concepts Demonstrated

This project demonstrates several important concepts in reinforcement learning and machine learning systems.

The project illustrates deep reinforcement learning by training neural networks through reward-based interaction with a simulated environment. It demonstrates policy gradient optimization by learning a stochastic policy that maximizes expected rewards. The system also shows how actor–critic methods can reduce training variance by combining policy learning with value estimation.

Additionally, it demonstrates how simulation environments can be used as training platforms for autonomous decision-making systems.

---

# Future Improvements

Several improvements could further enhance the system.

Possible extensions include implementing generalized advantage estimation, parallel environment sampling to increase training speed, entropy regularization to encourage exploration, automated hyperparameter tuning, and experiments with more complex continuous-control environments.

---

# Training

Training can be started by running the main agent script. During execution the system prints episode rewards, updates a live reward plot, and periodically saves model checkpoints.

---

# Evaluation

Trained models can be evaluated by running test episodes. The evaluation pipeline records gameplay videos of the trained agent interacting with the environment, allowing visual inspection of the learned landing behavior.

---

