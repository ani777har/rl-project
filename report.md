# Reinforcement Learning Final Project Report: Comparative Analysis of SAC, TD3, and PPO on InvertedDoublePendulum-v4

**Project Title:** Reinforcement Learning for Continuous Control: Balancing an Inverted Double Pendulum  
**Project:** InvertedDoublePendulum-v4 (Gymnasium/MuJoCo Environment)  
**Team Members:** Husik Sargsyan, Ani Harutyunyan, Viktoria Melkumyan  
**Instructor:** Davit Ghazaryan  
**Institution:** American University of Armenia  
**Date:** December 8, 2025  

---

## Table of Contents
1. [Problem Definition](#1-problem-definition)  
2. [Environment Description](#2-environment-description)  
3. [Algorithm Summary](#3-algorithm-summary)  
4. [Training Setup](#4-training-setup)  
5. [Results and Plots](#5-results-and-plots)  
6. [Observations and Analysis](#6-observations-and-analysis)  
7. [Challenges and Future Improvements](#7-challenges-and-future-improvements)  
8. [Conclusion](#8-conclusion)  

---

## 1. Problem Definition

### 1.1 Background on Reinforcement Learning
Reinforcement Learning (RL) is a subfield of machine learning where an agent learns to make decisions by interacting with an environment to maximize cumulative rewards. Unlike supervised learning, RL does not rely on labeled data; instead, it uses trial-and-error exploration to discover optimal policies. RL has achieved remarkable success in domains such as game playing (e.g., AlphaGo), robotics, autonomous vehicles, and resource management.

Continuous control problems, where actions and states are real-valued, pose unique challenges due to the infinite action space and the need for precise, smooth policies. Traditional RL algorithms like Q-learning are designed for discrete actions, necessitating adaptations for continuous domains.

### 1.2 The Inverted Double Pendulum Task
The inverted double pendulum is a classic benchmark in control theory and RL, representing an underactuated, nonlinear, and unstable dynamical system. It consists of a cart moving on a track with two hinged pendulum links attached. The goal is to apply forces to the cart to keep the pendulums upright, counteracting gravity and dynamic instabilities.

This task is particularly challenging because:
- It exhibits chaotic behavior with small perturbations leading to large deviations.
- It requires balancing multiple degrees of freedom simultaneously.
- It tests the agent's ability to handle delayed effects and long-term planning.

### 1.3 Significance and Objectives
This project evaluates three leading RL algorithms on the InvertedDoublePendulum-v4 environment from Gymnasium, powered by MuJoCo physics simulation. The objectives are:
- To compare algorithm performance in terms of convergence speed, final reward, and robustness.
- To analyze the impact of hyperparameter tuning on RL effectiveness.
- To provide insights into algorithm selection for continuous control tasks.

The experiment contributes to the broader RL literature by benchmarking modern algorithms on a standardized environment, aiding researchers and practitioners in algorithm selection.

---

## 2. Environment Description

### 2.1 Overview
The `InvertedDoublePendulum-v4` environment simulates a cart-pole system with two pendulum links using MuJoCo, a high-performance physics engine. MuJoCo enables realistic modeling of rigid body dynamics, collisions, and constraints, making it ideal for RL research. The environment is deterministic, ensuring reproducible experiments.

The system dynamics are governed by Newton's laws, with the cart's motion influencing the pendulums through the hinge. Episodes terminate if the pendulums fall below a threshold or exceed time limits.

### 2.2 State Space (Observations)
The state is a continuous 11-dimensional vector, providing full information about the system's configuration and velocities.

*   **Type**: Continuous (Box space).
*   **Shape**: `(11,)`
*   **Detailed Components**:
    1.  `cart_pos`: Horizontal position of the cart (meters). Indicates displacement from center.
    2.  `cart_vel`: Velocity of the cart (m/s). Reflects momentum.
    3.  `theta1`: Angle of the first pendulum link (radians). 0 when upright.
    4.  `theta1_dot`: Angular velocity of the first link (rad/s). Measures rotational speed.
    5.  `theta2`: Angle of the second pendulum link (radians). Relative to vertical.
    6.  `theta2_dot`: Angular velocity of the second link (rad/s).
    7.  `cos_theta1`: Cosine of `theta1`. Facilitates learning by providing periodic features.
    8.  `sin_theta1`: Sine of `theta1`. Completes the angle representation.
    9.  `cos_theta2`: Cosine of `theta2`.
    10. `sin_theta2`: Sine of `theta2`.
    11. `theta2 - theta1`: Relative angle between links. Important for coordination.
*   **Range**: Technically unbounded, but constrained by physical limits (e.g., angles wrap around).

These observations enable the agent to infer the system's state without partial observability issues.

### 2.3 Action Space
*   **Type**: Continuous (Box space).
*   **Shape**: `(1,)` (single scalar action).
*   **Range**: `[-3.0, 3.0]` Newtons. This force is applied horizontally to the cart.
*   **Interpretation**: Positive forces accelerate the cart right, negative left. The cart's motion exerts torques on the pendulums via the hinges, allowing indirect control.

The continuous action space requires algorithms capable of handling real-valued outputs, unlike discrete-action methods.

### 2.4 Reward Function
The reward is dense and shaped to encourage stability and efficiency:

$$ r_t = 10 \cdot (\cos(\theta_1) + \cos(\theta_2)) - 0.01 \cdot a_t^2 + 0.1 \cdot (1 - |\theta_2 - \theta_1|) $$

Where:
- $10 \cdot (\cos(\theta_1) + \cos(\theta_2))$: Bonus for upright pendulums (max 20 when both vertical).
- $-0.01 \cdot a_t^2$: Quadratic penalty for large actions, promoting energy efficiency.
- $0.1 \cdot (1 - |\theta_2 - \theta_1|)$: Encourages alignment between links.

Episodes yield cumulative rewards from ~0 (failure) to 10,000+ (success), with discounting (γ=0.99) used in training.

### 2.5 Episode Dynamics
- **Termination**: Pendulums fall (angles exceed thresholds) or max steps (typically 1000).
- **Reset**: Random initial conditions near upright position.
- **Difficulty**: Nonlinear dynamics make it hard to predict long-term effects.

---

## 3. Algorithm Summary

We evaluated three state-of-the-art RL algorithms for continuous control, implemented via Stable Baselines3.

### 3.1 Soft Actor-Critic (SAC)
SAC is an off-policy, maximum entropy RL algorithm that balances reward maximization with policy entropy for better exploration.

**Key Features**:
- Learns a stochastic policy π(a|s) and two Q-value critics.
- Objective: Maximize $J(\pi) = \mathbb{E}_{s \sim \rho^\pi, a \sim \pi} [ \sum_{t=0}^\infty \gamma^t (r_t + \alpha \log \pi(a_t|s_t)) ]$
- Uses replay buffer for sample efficiency.
- Temperature parameter α adapts exploration.

**Advantages**: Robust exploration, stable in continuous spaces.  
**Disadvantages**: Computationally intensive due to dual critics.

### 3.2 Twin Delayed Deep Deterministic Policy Gradient (TD3)
TD3 improves upon DDPG by addressing overestimation bias and enhancing stability.

**Key Features**:
- Learns deterministic policy μ(s) and twin critics.
- Updates policy less frequently (delayed) and adds noise to targets.
- Objective: Maximize Q-values via deterministic policy gradient.

**Advantages**: Stable convergence, effective for continuous control.  
**Disadvantages**: Deterministic policies may under-explore.

### 3.3 Proximal Policy Optimization (PPO)
PPO is an on-policy algorithm using trust-region optimization for stable updates.

**Key Features**:
- Learns stochastic policy and value function.
- Clipped surrogate objective: $L^{CLIP}(\theta) = \mathbb{E} [\min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t)]$
- Uses Generalized Advantage Estimation (GAE).

**Advantages**: Sample-efficient, easy to tune.  
**Disadvantages**: Requires fresh data per update.

All algorithms use MLP policies (256-256 units) and are trained with γ=0.99.

---

## 4. Training Setup

### 4.1 Hyperparameter Tuning
Hyperparameter tuning is crucial for RL performance. We used random search over 6 trials per algorithm, each with 75,000 timesteps, evaluating on 5 episodes.

**Search Spaces**:
- SAC: Learning rate (1e-5 to 3e-3), batch size (128,256,512), etc.
- TD3: Similar, with policy delay (1-3).
- PPO: N_steps (256-4096), clip_range (0.1-0.3), etc.

**Best Configurations** (from tuning):

| Parameter | SAC | TD3 | PPO |
|-----------|-----|-----|-----|
| Learning Rate | 0.000299 | 0.000475 | 0.000435 |
| Batch Size | 512 | 512 | 256 |
| Gamma | 0.9968 | 0.9805 | 0.9808 |
| Tau | 0.0098 | 0.0123 | N/A |
| Train Freq | 8 | 8 | N/A |
| Gradient Steps | 1 | 1 | N/A |
| Learning Starts | 10,000 | 5,000 | N/A |
| Policy Delay | N/A | 1 | N/A |
| N Steps | N/A | N/A | 2048 |
| N Epochs | N/A | N/A | 10 |
| GAE Lambda | N/A | N/A | 0.870 |
| Clip Range | N/A | N/A | 0.278 |

### 4.2 Final Training
Using best hyperparameters, we trained each algorithm for 500,000 timesteps with seed 0. Evaluation used deterministic rollouts over 20 episodes.

**Setup Details**:
- Environment: Monitored with Monitor wrapper.
- Logging: TensorBoard for metrics, episode rewards via callback.
- Hardware: Standard CPU/GPU setup.

---

## 5. Results and Plots

### 5.1 Performance Metrics
Final evaluation results:

| Algorithm | Mean Reward | Std Dev | Training Time (s) | Episodes Logged |
|-----------|-------------|---------|-------------------|-----------------|
| SAC       | 9354.95     | ±0.10   | 465.11            | 3831            |
| TD3       | 9358.38     | ±1.45   | 486.14            | 3329            |
| PPO       | 9358.90     | ±0.14   | 96.58             | 8291            |

All algorithms achieved near-optimal performance (~9350+ rewards).

### 5.2 Learning Curves
![learning_curves](learning_curves.png)
Curves show episode rewards over time. PPO converged fastest in wall-clock time, while SAC/TD3 leveraged replay buffers for efficiency.

### 5.3 Visualizations

## SAC
![sac_gif](videos/sac_final.gif)
## TD3
![td3_gif](videos/td3_final.gif)
## PPO
![ppo_gif](videos/ppo_final.gif)
---

## 6. Observations and Analysis

### 6.1 Algorithm Comparison
- **Performance**: PPO slightly outperformed others, likely due to on-policy stability.
- **Efficiency**: PPO was 5x faster than off-policy methods, despite higher episode counts.
- **Robustness**: Low std devs indicate reliable policies.

### 6.2 Insights
- Hyperparameter tuning was essential; default settings underperformed.
- Off-policy methods excel in sample efficiency but require more computation.
- The environment's simplicity favored all algorithms equally.

---

## 7. Challenges and Future Improvements

### 7.1 Challenges
- Hyperparameter sensitivity.
- Computational costs of tuning.
- Balancing exploration vs. exploitation.

### 7.2 Future Work
- Extend to more complex environments.
- Implement advanced tuning (e.g., Bayesian optimization).
- Add robustness tests with perturbations.

---

## 8. Conclusion
This project demonstrated the effectiveness of SAC, TD3, and PPO on InvertedDoublePendulum-v4, with PPO offering the best wall-clock efficiency. Results highlight the importance of tuning and provide a foundation for further RL research.
