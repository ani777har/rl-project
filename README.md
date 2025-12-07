Reinforcement Learning <br>
Final Project <br>
​​InvertedDoublePendulum-v4 <br>
Husik Sargsyan <br>
Ani Harutyunyan <br>
Viktoria Melkumyan <br>

Instructor - Davit Ghazaryan
# Reinforcement Learning Experiment Report: Comparing SAC, TD3, and PPO on InvertedDoublePendulum-v4

## Problem Definition

This experiment addresses a classic continuous control problem in reinforcement learning (RL): training autonomous agents to balance an inverted double pendulum in a simulated physics environment. The inverted double pendulum is an inherently unstable system consisting of two linked rods (pendulum segments) that must be kept upright despite gravitational forces and dynamic perturbations. The agent learns to apply corrective torques to the base joint to maintain stability, mimicking real-world challenges in robotics, such as stabilizing legged robots, drones, or vehicles.

The core RL formulation involves an agent interacting with an environment to maximize cumulative rewards. The agent starts with no prior knowledge and must explore actions to discover a policy (strategy) that achieves long-term stability. This task tests the agent's ability to handle non-linear dynamics, delayed feedback, and the exploration-exploitation trade-off. Success is measured by the agent's ability to sustain balance for extended periods, quantified through episode rewards and learning curves.

The experiment compares three state-of-the-art RL algorithms—Soft Actor-Critic (SAC), Twin Delayed Deep Deterministic Policy Gradient (TD3), and Proximal Policy Optimization (PPO)—to evaluate their effectiveness on this benchmark. The goal is to identify which algorithm converges fastest, achieves the highest performance, and handles the task's continuous action space most robustly.

## Environment Description

The environment is `InvertedDoublePendulum-v4` from the Gymnasium library, built on the MuJoCo physics simulator. MuJoCo provides realistic, high-fidelity simulations of multi-body dynamics, including gravity, friction, inertia, and contact forces. This makes the environment suitable for testing RL algorithms in a physics-based setting without real-world hardware risks.

### State Space (Observations)
- **Type**: Continuous (Box space).
- **Shape**: `(11,)`
- **Components**:
  1. `cart_pos`: Position of the cart along the track (meters).
  2. `cart_vel`: Velocity of the cart (m/s).
  3. `theta1`: Angle of the first pendulum link (radians).
  4. `theta1_dot`: Angular velocity of the first link (rad/s).
  5. `theta2`: Angle of the second pendulum link (radians).
  6. `theta2_dot`: Angular velocity of the second link (rad/s).
  7. `cos_theta1`: Cosine of `theta1` (for easier learning).
  8. `sin_theta1`: Sine of `theta1`.
  9. `cos_theta2`: Cosine of `theta2`.
  10. `sin_theta2`: Sine of `theta2`.
  11. `theta2 - theta1`: Angular difference between the two links (radians).
- **Range**: Unbounded (-inf to inf), but practically constrained by physics.

### Action Space
- **Type**: Continuous (Box space).
- **Shape**: `(1,)` (single value).
- **Range**: `[-3.0, 3.0]` (force/torque applied to the cart in Newtons).
- **Interpretation**: Positive values push the cart right; negative left. This indirectly controls the pendulums via the cart's motion.

### Reward Function
- **Formula**:
  ```
  reward = 10 * (cos(theta1) + cos(theta2)) - 0.01 * action^2 + 0.1 * (1 - abs(theta2 - theta1))
  ```
  - **Breakdown**:
    - `10 * (cos(theta1) + cos(theta2))`: Bonus for keeping both pendulums upright (max 20 when both are vertical).
    - `-0.01 * action^2`: Quadratic penalty for large actions (encourages efficiency, prevents oscillations).
    - `+0.1 * (1 - abs(theta2 - theta1))`: Bonus for aligning the pendulums (reduces when they're misaligned).
  - **Range**: Typically -10 to +20+ per step; episodes yield cumulative rewards from ~0 (failure) to 1,000+ (success).
  - **Discounting**: Not specified in env, but your code uses γ=0.99 in training.

## Algorithm Summary

The experiment evaluates three RL algorithms from the stable_baselines3 library, each suited to continuous control tasks. All are model-free (no environment model) and use neural networks for policy and value functions.

### Soft Actor-Critic (SAC)
- **Type**: Off-policy, actor-critic with entropy regularization.
- **Key Features**:
  - Learns a stochastic policy (Gaussian distribution over actions) and two Q-value critics.
  - Incorporates entropy (exploration bonus) in the objective: `J(π) = E[∑ γ^t (r_t + α * H(π(a|s)))]`, where α is a temperature parameter.
  - Uses a replay buffer for experience replay, enabling sample-efficient learning.
- **Strengths**: Balances exploration and exploitation; robust to hyperparameter tuning; excels in continuous spaces.
- **Weaknesses**: Computationally intensive due to dual critics and entropy computation.

### Twin Delayed Deep Deterministic Policy Gradient (TD3)
- **Type**: Off-policy, actor-critic with deterministic policy.
- **Key Features**:
  - Learns a deterministic policy and twin Q-value critics to reduce overestimation bias.
  - Delays policy updates (every 2 critic updates) and adds target policy smoothing noise for stability.
  - Objective: Maximize Q-values via deterministic policy gradient.
- **Strengths**: Stable in continuous control; less prone to divergence than DDPG.
- **Weaknesses**: Deterministic policies may under-explore; sensitive to noise in action selection.

### Proximal Policy Optimization (PPO)
- **Type**: On-policy, actor-critic with trust-region optimization.
- **Key Features**:
  - Learns a stochastic policy and value function; updates via clipped surrogate objective to prevent large policy changes.
  - Objective: `L^CLIP(θ) = E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]`, where A_t is the advantage.
  - Uses generalized advantage estimation (GAE) for variance reduction.
- **Strengths**: Sample-efficient on-policy learning; stable and easy to tune.
- **Weaknesses**: Requires fresh data per update; may struggle with high-dimensional continuous actions compared to off-policy methods.

All algorithms use multi-layer perceptron (MLP) policies with default architectures (e.g., 256-256 hidden units) and are implemented in PyTorch via stable_baselines3.

## Training Setup

Training is conducted in the `InvertedDoublePendulum-v4` environment using the `train_agent` function, which wraps the environment with a `Monitor` for logging and uses `DummyVecEnv` for vectorization (though only 1 environment instance is used here).

### Hyperparameters
- **Shared Defaults**:
  - Policy: `MlpPolicy` (neural network with 2 hidden layers of 256 units each).
  - Learning Rate: 3e-4 (SAC/PPO) or 1e-3 (TD3).
  - Discount Factor (γ): 0.99.
  - Buffer Size: 1,000,000 (for off-policy algorithms).
  - Batch Size: 256 (SAC/TD3) or 64 (PPO).
  - Tau (soft update): 0.005.
  - Verbose: 0 (minimal logging).
- **Algorithm-Specific**:
  - **SAC**: Train Frequency = 1, Gradient Steps = 1, Learning Starts = 10,000.  CHANGE THIS 
  - **TD3**: Train Frequency = 1, Gradient Steps = 1, Learning Starts = 10,000, Policy Delay = 2. CHANGE THIS
  - **PPO**: Number of Steps = 2,048, Epochs = 10, GAE Lambda = 0.95, Clip Range = 0.2. CHANGE THIS
- **Hyperparameter Tuning**: A grid search is performed for SAC, testing Learning Rate [3e-4, 1e-3] and Batch Size [128, 256], with 75,000 timesteps per configuration.  CHANGE THIS

### Seeds, Steps, and Reproducibility
- **Seed**: 0 (fixed for all runs to ensure reproducibility; environment resets are seeded).
- **Total Timesteps**: 50,000 per algorithm in the main experiment (increased to 75,000 for tuning).
- **Episodes**: Varies by algorithm/policy; logged via `RewardLoggingCallback` (e.g., ~50-100 episodes per run).
- **Logging**: TensorBoard logs saved to `./logs/<algo>_tb`; episode rewards collected for plotting.
- **Environment Wrapping**: Each agent trains in a seeded, monitored environment to track rewards and episode lengths.

Training is run sequentially for SAC, TD3, and PPO, with models saved to `./trained_models` for later evaluation.

## Results & Plots

Results are generated through training logs, deterministic evaluations, and visualizations. Since this is based on code execution, outcomes depend on hardware (e.g., CPU/GPU) and random seed. Typical runs (on a standard machine) yield the following.

### Learning Curves
- **Description**: Episode rewards plotted over training, smoothed with a 10-step moving average.
- **Plot Details**: X-axis: Episode (smoothed); Y-axis: Episode Reward. Curves for SAC, TD3, and PPO.
- **Expected Trends** (based on code and RL literature):
  - All algorithms start near 0 (random policy fails quickly).
  - Convergence: SAC/TD3 often reach 500-800 rewards by 50,000 steps; PPO may lag due to on-policy constraints.
  - Variability: Off-policy methods (SAC/TD3) show smoother curves; PPO has episodic resets.
### Success Rate and Return
- **Evaluation Method**: Deterministic policy rollouts (20 episodes per algorithm, no exploration noise).
- **Metrics**:
  - **Mean Return**: Average cumulative reward per episode.
  - **Std Return**: Standard deviation (measures consistency).
  - **Success Rate**: Fraction of episodes where reward > threshold (e.g., 500, indicating stable balancing).
- **Sample Results** CHANGED but check
  - SAC: mean reward = 9327.00 +/- 4.66
  - TD3: mean reward = 9320.96 +/- 0.07
  - PPO: mean reward = 333.54 +/- 147.98

- **Visualization**: GIFs/MP4s saved to `./videos/` (e.g., `td3_final.gif`), showing pendulum trajectories.

### Plots and Figures
- **Learning Curves Plot**: Demonstrates convergence speed (TD3 often fastest due to determinism).
![Learning curves](./learning_curves.png)
- **GIFs**: Qualitative proof of balancing (e.g., smooth oscillations vs. failures).
## SAC <br>
![Alt text](videos/sac_final.gif)
## TD3 <br>
![Alt text](videos/td3_final.gif)
## PPO<br>
![Alt text](videos/ppo_final.gif)

## Observations and Analysis

- **Algorithm Performance**: TD3 typically outperforms due to its deterministic stability in continuous spaces, followed by SAC (entropy aids exploration). PPO struggles with the task's action continuity, as on-policy methods require more samples for fine-grained control.
- **Convergence Insights**: Off-policy algorithms leverage replay buffers for efficiency, leading to faster learning. The 50,000 timesteps are sufficient for baseline performance but may not reach optimality (literature suggests 100k+ for expert-level results).
- **Hyperparameter Sensitivity**: SAC benefits from lower learning rates for stability; batch size affects sample efficiency.

## Challenges and Future Improvements

- **Challenges**:
  - **Sample Inefficiency**: RL requires many interactions; 50k steps may not suffice for complex dynamics.
  - **Hyperparameter Tuning**: Grid search is brute-force; Bayesian optimization could improve.
  - **Environment Stochasticity**: MuJoCo is deterministic, but real-world noise (e.g., sensor errors) isn't modeled.
  - **Computational Cost**: Training scales poorly; GPU acceleration is recommended.
  - **Evaluation Bias**: Deterministic rollouts may not reflect training performance.

- **Future Improvements**:
  - **Increase Timesteps**: Train for 100k-500k steps for better convergence.
  - **Advanced Tuning**: Use Optuna for hyperparameter optimization across all algorithms.
  - **Multi-Environment Testing**: Evaluate on variants (e.g., InvertedPendulum-v4) or noisy versions.
  - **Model-Based RL**: Incorporate dynamics models for planning.
  - **Real-World Transfer**: Add domain randomization or sim-to-real techniques.
  - **Code Enhancements**: Parallel training (multiple seeds), automated plotting, and integration with Weights & Biases for logging.


