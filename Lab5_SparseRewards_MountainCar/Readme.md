# Lab 5 — Sparse Reward Reinforcement Learning: DQN, Reward Shaping, Curriculum Learning & PPO on MountainCar

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Required-orange)
![Gymnasium](https://img.shields.io/badge/Gymnasium-Required-green)
![StableBaselines3](https://img.shields.io/badge/StableBaselines3-Required-lightblue)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Required-green)
![Field](https://img.shields.io/badge/Field-Robotics%20&%20AI-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

> **Course:** Machine Learning for Robotics — Faculty of Control Systems and Robotics, ITMO University <br>
> **Author:** Umer Ahmed Baig Mughal — MSc Robotics and Artificial Intelligence <br>
> **Topic:** DQN · Sparse Reward · Reward Shaping · Reverse Curriculum Learning · Hindsight Experience Replay · PPO · MountainCar

---

## Table of Contents

1. [Objective](#objective)
2. [Theoretical Background](#theoretical-background)
   - [The Sparse Reward Problem](#the-sparse-reward-problem)
   - [Deep Q-Network (DQN)](#deep-q-network-dqn)
   - [Experience Replay](#experience-replay)
   - [Reward Shaping](#reward-shaping)
   - [Reverse Curriculum Learning](#reverse-curriculum-learning)
   - [Hindsight Experience Replay (HER)](#hindsight-experience-replay-her)
   - [Proximal Policy Optimisation (PPO)](#proximal-policy-optimisation-ppo)
   - [System Properties](#system-properties)
3. [Tasks and Implementation](#tasks-and-implementation)
   - [Task 1 — Vanilla DQN (Baseline Failure)](#task-1--vanilla-dqn-baseline-failure)
   - [Task 2 — Energy-Based Reward Shaping](#task-2--energy-based-reward-shaping)
   - [Task 3 — Reverse Curriculum Learning](#task-3--reverse-curriculum-learning)
   - [Task 4 — PPO on MountainCar Continuous](#task-4--ppo-on-mountaincar-continuous)
4. [Validation and Metrics](#validation-and-metrics)
5. [System Parameters](#system-parameters)
6. [Implementation](#implementation)
   - [File Structure](#file-structure)
   - [Class Reference](#class-reference)
   - [Algorithm Walkthrough](#algorithm-walkthrough)
7. [How to Run](#how-to-run)
8. [Results](#results)
9. [Analysis and Discussion](#analysis-and-discussion)
10. [Dependencies](#dependencies)
11. [Notes and Limitations](#notes-and-limitations)
12. [Author](#author)
13. [License](#license)

---

## Objective

This lab investigates and solves the **sparse reward problem** in reinforcement learning using the MountainCar benchmark environments from OpenAI Gymnasium. The core challenge: standard DQN with the default reward signal cannot solve `MountainCar-v0` because the agent almost never reaches the goal by random exploration, receiving a constant penalty of −1 per step with no gradient signal toward success.

The lab implements and compares four progressively more sophisticated approaches to overcome sparse rewards:

- Implementing a **vanilla DQN with experience replay** from scratch in PyTorch, establishing that it fails to solve MountainCar within 150 episodes due to the reward sparsity, and understanding why.
- Designing an **energy-based reward shaping function** that augments the sparse reward with dense feedback derived from the car's total mechanical energy (kinetic + potential), enabling the DQN agent to solve the task within 400 episodes.
- Implementing **Reverse Curriculum Learning** — progressively training the agent from starting states increasingly far from the goal — as a structured exploration strategy that achieves near-perfect success rates within 1000 episodes with the original sparse reward.
- Solving **MountainCarContinuous-v0** using **PPO** (Proximal Policy Optimisation) from Stable Baselines 3, justifying the algorithm choice against alternatives (TD3, SAC, TRPO) and achieving near-optimal performance (reward ≈ 0) in 200 000 timesteps.

The lab is implemented as a single Google Colab Jupyter notebook (`Sparse_Reward_RL_DQN_PPO_Curriculum_Learning_MountainCar.ipynb`) and produces training reward curves, loss curves, success rate plots, and evaluation results for each method.

---

## Theoretical Background

### The Sparse Reward Problem

In dense reward settings, an agent receives informative feedback at nearly every timestep, making gradient-based policy improvement tractable. In **sparse reward** settings, a reward signal is only delivered when the agent achieves a specific goal — which may require a long, precisely-coordinated sequence of actions to reach by chance.

`MountainCar-v0` is a canonical sparse reward benchmark:

```
State:       s = (position, velocity) ∈ ℝ²
Actions:     a ∈ {0 (push left), 1 (no push), 2 (push right)}
Reward:      r = −1  at every step (no goal bonus in base environment)
Terminal:    position ≥ 0.5  (goal flag reached)  OR  t = 200 (timeout)
```

The car starts in a valley and must build momentum by rocking back and forth — a multi-step strategy that a random policy essentially never discovers. Without ever seeing a success, Q-values never receive a positive update signal, and the agent remains stuck.

### Deep Q-Network (DQN)

DQN approximates the action-value function $Q(s, a)$ with a neural network parameterised by $\theta$, trained to minimise the Bellman residual:

```
L(θ) = 𝔼 [ (r + γ max_{a'} Q(s', a'; θ̄) − Q(s, a; θ))² ]

where:
    θ̄   — target network parameters (periodically copied from θ)
    γ   — discount factor
    r   — observed reward
    s'  — next state
```

In this lab the DQN network is a fully-connected MLP:

```
Input(2) → Linear(2→h) → ReLU → Linear(h→2h) → ReLU → Linear(2h→3)

where h is the hidden dimension (12 in Task 1, 32–64 in Tasks 2–3)
Output: Q-value estimates for each of the 3 discrete actions
```

Training uses the **Adam optimiser** with MSE loss. The ε-greedy policy selects random actions with probability ε (decaying over training) and the greedy action otherwise.

### Experience Replay

To break temporal correlations between consecutive transitions and improve sample efficiency, transitions `(s, a, s', r, done)` are stored in a **replay buffer** and sampled randomly for each training update:

```python
memory.append((state, action, next_state, reward, done))
batch = random.sample(memory, batch_size)

# Bellman target computation
targets = rewards + γ · max_a' Q(next_states, a')
targets[done_indices] = rewards[done_indices]    # no bootstrap at terminal
```

Without experience replay, the network would overfit to the most recent transitions and exhibit catastrophic forgetting of earlier experiences.

### Reward Shaping

Reward shaping augments the environment reward with an auxiliary signal $F(s, s')$ that guides exploration without changing the optimal policy, provided the shaping is **potential-based**:

```
r_shaped(s, a, s') = r_env(s, a, s') + F(s, s')

Potential-based shaping:  F(s, s') = γ · Φ(s') − Φ(s)

where Φ(s) is the potential function (any scalar state function)
```

In this lab, the **total mechanical energy** of the car is used as the potential:

```
E(s) = g · (pos − pos_min)  +  0.5 · vel²
         └───────────────┘    └──────────┘
         Potential Energy      Kinetic Energy

F(s, s') = k_energy · (E(s') − E(s))    where k_energy = 90.0
```

This gives the agent a positive signal whenever it gains mechanical energy (builds momentum toward the goal) and a negative signal when it loses energy, providing dense gradient information even without reaching the flag.

### Reverse Curriculum Learning

Reverse Curriculum Learning (also known as Backwards Curriculum) initialises training from **states close to the goal** and progressively expands the start state distribution outward. The intuition is:

```
Phase 1 (easy):   start near the goal → agent quickly discovers success
Phase 2:          start slightly farther → agent now knows what success looks like
...
Phase N (full):   start from the standard initial distribution
```

This ensures the agent always receives enough positive reward signal to learn from, regardless of how far the initial state is from the goal. The curriculum avoids the cold-start failure mode of uniform initialisation on sparse tasks.

### Hindsight Experience Replay (HER)

HER addresses sparse rewards by replaying failed trajectories **as if the agent had been trying to reach the state it actually ended up in**. For a trajectory that ended at state $s_T$ (not the intended goal $g$), HER creates synthetic transitions with a substitute goal $g' = s_T$ and retroactively assigns positive reward:

```
Original transition:  (s_t, a_t, s_{t+1}, r=-1, g=0.5)    [failure]
HER relabelled:       (s_t, a_t, s_{t+1}, r=0,  g=s_T)    [success — in hindsight]
```

By augmenting the replay buffer with these relabelled transitions, the agent learns a goal-conditioned policy even from trajectories that never reached the intended goal. HER is referenced in this lab and can be implemented via the `HerReplayBuffer` in Stable Baselines 3.

### Proximal Policy Optimisation (PPO)

PPO is an on-policy policy gradient algorithm that constrains the policy update magnitude at each step using a clipped surrogate objective:

```
L^CLIP(θ) = 𝔼 [ min( r_t(θ) Â_t ,  clip(r_t(θ), 1−ε, 1+ε) Â_t ) ]

where:
    r_t(θ)  = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)    (probability ratio)
    Â_t     = generalised advantage estimate (GAE)
    ε       = 0.2                                         (clip range)
```

The clipping prevents excessively large policy updates, improving stability compared to vanilla policy gradient methods. PPO supports **continuous action spaces** natively, making it suitable for `MountainCarContinuous-v0` where the action is a continuous force in `[−1, 1]`.

### System Properties

| Property | Value | Notes |
|----------|-------|-------|
| Discrete environment | `MountainCar-v0` | gymnasium classic control |
| Continuous environment | `MountainCarContinuous-v0` | gymnasium classic control |
| State space | 2D — (position, velocity) | position ∈ [−1.2, 0.6], velocity ∈ [−0.07, 0.07] |
| Discrete action space | 3 — {left, neutral, right} | integer actions |
| Continuous action space | 1D force — [−1, 1] | real-valued control |
| Default reward | −1 per step | No explicit goal bonus |
| Success condition | position ≥ 0.5 | Car reaches the flag |
| Episode time limit | 200 steps | Truncation, not termination |
| DQN network depth | 3 layers | Input → h → 2h → output |
| Optimiser | Adam | Tasks 1–3 |
| Loss function | MSE | Bellman residual |
| PPO backend | Stable Baselines 3 | `MlpPolicy` |
| Hardware | CPU (Google Colab) | No GPU required |

---

## Tasks and Implementation

### Task 1 — Vanilla DQN (Baseline Failure)

**Goal:** Implement DQN with experience replay and demonstrate that it **cannot** solve `MountainCar-v0` with the default sparse reward within 150 episodes.

```python
class DQN():
    def __init__(self, state_dim, action_dim, hidden_dim=12, alpha=0.001):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 2 * hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * hidden_dim, action_dim)
        )
        self.optimizer = torch.optim.Adam(lr=alpha, params=self.model.parameters())
        self.criterion = torch.nn.MSELoss()

    def replay(self, memory, size, tmodel, gamma):
        batch = random.sample(memory, size)
        # Bellman backup with target model
        all_q_values[range(len(all_q_values)), actions] = (
            rewards + gamma * torch.max(all_q_values_next, axis=1).values
        )
        all_q_values[done_indices, actions[done_indices]] = rewards[done_indices]
        self.update(states.tolist(), all_q_values.tolist())
```

Key parameters: `hidden_dim=12`, `alpha=0.001`, `epsilon=0.9` (decays by factor 0.6 every 10 episodes after episode 50), `replay_size=10`, `EPISODES_MAX=150`, `STEPS_MAX=200`. The agent consistently fails to reach the flag, logging a total reward of −200 (full timeout) in every episode due to insufficient exploration and no positive reinforcement.

---

### Task 2 — Energy-Based Reward Shaping

**Goal:** Design and implement an energy-based auxiliary reward function that provides dense feedback, enabling DQN to solve `MountainCar-v0` within 400 episodes.

```python
g = 9.8
pos_min = env.observation_space.low[0]    # −1.2  (bottom of the valley)
k_energy = 90.0

def energy(s):
    pos, vel = s
    potential = g * (pos - pos_min)
    kinetic   = 0.5 * vel ** 2
    return potential + kinetic

# Per-step shaped reward
E_prev = energy(state)
# ... step ...
E_next = energy(next_state)
delta_E = E_next - E_prev
reward_shaped = orig_r + k_energy * delta_E

# Bonus on reaching the goal
if done and next_state[0] >= 0.5:
    reward_shaped += 50.0
```

The shaped reward augments the constant −1 with a term proportional to the change in mechanical energy, scaled by `k_energy=90.0`. Positive energy gain (car gaining speed or climbing) gives a positive bonus; energy loss gives a penalty. The goal-reaching bonus of +50 provides an additional terminal incentive. Training uses `EPISODES=400`, `replay_size=32`, `eps_decay=0.995`, `memory_limit=50000`.

---

### Task 3 — Reverse Curriculum Learning

**Goal:** Implement Reverse Curriculum Learning using the original sparse reward, without any reward shaping, and demonstrate that it achieves higher success rates and more stable training than reward shaping.

```python
# TRAINING LOOP — DQN with Reverse Curriculum
EPISODES      = 1000
BATCH_SIZE    = 32
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN   = 0.01
MEMORY_LIMIT  = 50000

class DQN:
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=5e-4):
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, action_dim)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def replay(self, memory, batch_size, gamma):
        # Double-network Bellman backup
        targets = rewards + gamma * max_next_q * (~dones)
        q_targets[range(batch_size), actions] = targets
        self.update(states, q_targets)
```

The curriculum progressively widens the initial state distribution: early episodes start near the goal, and the starting region expands as the agent accumulates success experience. Training logs episode reward, success rate (rolling 100-episode window), and DQN loss across 1000 episodes with live visualisation.

---

### Task 4 — PPO on MountainCar Continuous

**Goal:** Solve `MountainCarContinuous-v0` (continuous action space, sparse reward) using PPO from Stable Baselines 3 and justify the algorithm choice.

```python
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

log_dir = "ppo_mountain_logs/"
env = gym.make("MountainCarContinuous-v0")
env = Monitor(env, log_dir)

model = PPO(
    policy      = "MlpPolicy",
    env         = env,
    learning_rate = 3e-4,
    n_steps     = 2048,
    batch_size  = 64,
    gamma       = 0.99,
    gae_lambda  = 0.95,
    clip_range  = 0.2,
    ent_coef    = 0.01,
    verbose     = 1,
)

model.learn(total_timesteps=200_000)
model.save("ppo_mountaincar_continuous")
```

Post-training evaluation runs 10 deterministic episodes with `model.predict(obs, deterministic=True)` and plots per-episode reward. The test episode achieves a total reward of approximately −0.00017 — effectively optimal for this benchmark.

---

## Validation and Metrics

### Task 1 — DQN Failure Confirmation

| Criterion | Expected Outcome |
|-----------|-----------------|
| Episode reward | −200 (timeout) every episode — agent never reaches flag |
| Loss curve | Fluctuating, no consistent decrease — no positive Q-value signal |
| Success rate | 0% across all 150 episodes |

### Task 2 — Reward Shaping Success

| Metric | Description |
|--------|-------------|
| Original reward | Unmodified −1/step total; logged separately from shaped reward for honest comparison |
| Shaped reward | Augmented reward including energy bonus and goal bonus; expected to grow over training |
| Running average (50 ep) | Smoothed shaped reward trend — should show clear upward trajectory within 400 episodes |
| Convergence criterion | Agent consistently reaches `position ≥ 0.5` before step 200 |

### Task 3 — Curriculum Success Rate

| Metric | Description |
|--------|-------------|
| Episode reward | Sparse −1/step; no artificial augmentation |
| Success flag | `terminated and next_state[0] >= 0.5` |
| Rolling success rate (100 ep) | Primary convergence signal — target ≥ 80% success |
| Moving average reward (50 ep) | Smoothed reward trend overlaid on raw episode rewards |
| DQN loss | Per-update MSE loss; expected to decrease and stabilise |

### Task 4 — PPO Evaluation

| Metric | Target |
|--------|--------|
| Learning curve (timesteps vs. reward) | Monotonic increase from ≈ −20 to ≈ 0 |
| Evaluation reward mean ± std | Tightly clustered around 0 |
| Test episode total reward | ≈ 0 (success) |

---

## System Parameters

### Environment Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Discrete env | `MountainCar-v0` | Gymnasium classic control, v4 API |
| Continuous env | `MountainCarContinuous-v0` | Continuous action space variant |
| State dimension | 2 | (position, velocity) |
| Discrete action count | 3 | Push left, no push, push right |
| Continuous action range | [−1, 1] | Engine force |
| Success threshold | position ≥ 0.5 | Goal flag position |
| Steps per episode | 200 | Maximum before truncation |
| Initial position range | [−0.6, −0.4] | Uniformly sampled at reset |
| Initial velocity | 0.0 | Stationary at start |

### DQN Parameters (Tasks 1–3)

| Parameter | Task 1 | Task 2 | Task 3 | Description |
|-----------|:------:|:------:|:------:|-------------|
| Hidden dim | 12 | 32 | 64 | Neurons in first hidden layer |
| Learning rate α | 0.001 | 0.0025 | 5e-4 | Adam step size |
| Episodes | 150 | 400 | 1000 | Training episodes |
| Steps per episode | 200 | 200 | 200 | Truncation limit |
| Initial ε | 0.9 | 1.0 | 1.0 | Starting exploration rate |
| ε decay | ×0.6 per 10 ep | ×0.995 per step | ×0.995 per step | Decay schedule |
| ε minimum | 0.01 | 0.02 | 0.01 | Floor value |
| Replay buffer size | — | 50 000 | 50 000 | Memory capacity |
| Batch size | 10 | 32 | 32 | Transitions per update |
| Discount γ | 0.99 | 0.99 | 0.99 | Future reward discount |

### Reward Shaping Parameters (Task 2)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Energy scale k_energy | 90.0 | Energy delta multiplier |
| Gravitational constant g | 9.8 | m/s² (matches environment physics) |
| Position baseline pos_min | −1.2 | Bottom of the valley (observation low) |
| Goal bonus | +50.0 | Awarded when `position ≥ 0.5` on termination |
| Potential function Φ(s) | E(s) = g·(pos−pos_min) + 0.5·vel² | Total mechanical energy |

### PPO Parameters (Task 4)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Policy | MlpPolicy | Fully-connected actor-critic |
| Learning rate | 3e-4 | Adam step size |
| n_steps | 2048 | Steps per rollout before update |
| Batch size | 64 | Mini-batch size for gradient updates |
| Discount γ | 0.99 | Future reward discount |
| GAE λ | 0.95 | Generalised advantage estimation factor |
| Clip range ε | 0.2 | PPO surrogate objective clip |
| Entropy coefficient | 0.01 | Exploration bonus weight |
| Total timesteps | 200 000 | Total environment interaction budget |
| Eval episodes | 10 | Deterministic post-training evaluation |

---

## Implementation

### File Structure

```
Lab5/
├── Readme.md
└── Sparse_Reward_RL_DQN_PPO_Curriculum_Learning_MountainCar.ipynb
```

**Saved model outputs (generated during training):**

| File | Generated By | Description |
|------|-------------|-------------|
| `mountaincar_dqn_no_wrapper.pth` | Task 3 | PyTorch state dict of trained DQN |
| `ppo_mountaincar_continuous.zip` | Task 4 | Stable Baselines 3 saved PPO model |
| `ppo_mountain_logs/monitor.csv` | Task 4 | Training episode rewards logged by Monitor wrapper |
| `ppo_mountaincar_eval_plot.png` | Task 4 | Evaluation reward plot |

---

### Class Reference

#### `DQN` — Deep Q-Network agent (Tasks 1–3)

A PyTorch-based DQN agent with a 3-layer MLP Q-network, experience replay, and ε-greedy action selection.

```python
class DQN:
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=5e-4)
    def update(self, states_tensor, target_q_tensor)
    def predict(self, state) → torch.Tensor          # Q-values for all actions
    def replay(self, memory, batch_size, gamma)       # Sample-and-update from buffer
```

| Method | Purpose |
|--------|---------|
| `__init__` | Initialises MLP, Adam optimiser, MSE criterion, loss log |
| `update` | Forward pass, loss computation, backprop, weight update |
| `predict` | No-grad forward pass returning Q-values for state |
| `replay` | Sample mini-batch from replay buffer, compute Bellman targets, call `update` |

---

#### `energy(s)` — Mechanical energy potential function (Task 2)

Computes the total mechanical energy of the MountainCar state, used as the reward shaping potential Φ(s).

```python
def energy(s):
    pos, vel = s
    potential = g * (pos - pos_min)   # gravitational potential energy
    kinetic   = 0.5 * vel ** 2        # kinetic energy
    return potential + kinetic
```

| Argument | Type | Description |
|----------|------|-------------|
| `s` | `array-like` of shape (2,) | State vector (position, velocity) |

**Returns:** `float` — total mechanical energy at state `s`.

---

#### `update_plots(axs, rewards, successes, losses)` — Live training visualiser (Task 3)

Updates a 3-panel live plot during training: episode rewards with 50-episode moving average, rolling success rate (100-episode window), and DQN loss.

```python
def update_plots(axs, rewards, successes, losses):
    # Panel 1: episode reward + 50-episode moving average
    # Panel 2: 100-episode rolling success rate
    # Panel 3: last 1000 DQN update losses
```

---

#### `train_dqn_curriculum_free()` — Curriculum training loop (Task 3)

Full training loop implementing the DQN agent under a reverse curriculum schedule. Returns the trained agent, episode reward history, and success history.

```python
def train_dqn_curriculum_free() → (DQN, list, list):
    # Initialises env, agent, memory buffer
    # Trains for EPISODES with ε-greedy exploration + experience replay
    # Logs rewards, successes; calls update_plots every 50 episodes
    # Returns agent, episode_rewards, success_history
```

---

### Algorithm Walkthrough

**Complete pipeline (`Sparse_Reward_RL_DQN_PPO_Curriculum_Learning_MountainCar.ipynb`):**

```
1. Environment setup:
       pip install gymnasium[classic-control] pyvirtualdisplay
       apt-get install -y xvfb python-opengl ffmpeg
       Configure virtual display via pyvirtualdisplay.Display

2. TASK 1 — VANILLA DQN BASELINE:
       a. Initialise MountainCar-v0 environment
       b. Construct DQN(2, 3, hidden_dim=12, alpha=0.001)
       c. Run ε-greedy training for 150 episodes, STEPS_MAX=200
       d. Decay ε by factor 0.6 every 10 episodes after episode 50
       e. Store transitions in memory; update via model.replay(memory, 10, model, 0.99)
       f. Plot live loss curve → confirm total reward stays at −200 throughout

3. TASK 2 — ENERGY-BASED REWARD SHAPING:
       a. Redefine energy() function using gravitational + kinetic energy
       b. Construct DQN(2, 3, hidden_dim=32, alpha=0.0025)
       c. At each step: compute delta_E = energy(next_state) − energy(state)
          reward_shaped = orig_r + 90.0 * delta_E
          if done and goal reached: reward_shaped += 50.0
       d. Maintain memory buffer (capacity 50 000); batch updates every step
       e. Run 400 episodes with ε starting at 1.0, decaying by 0.995 each episode
       f. Log orig reward and shaped reward separately; plot both + running avg

4. TASK 3 — REVERSE CURRICULUM LEARNING:
       a. Construct larger DQN(2, 3, hidden_dim=64, lr=5e-4)
       b. Implement curriculum: progressively widen start state distribution from goal
       c. Run 1000 episodes with original sparse reward (no shaping)
       d. Track success flag: terminated and next_state[0] >= 0.5
       e. Update plots every 50 episodes via update_plots()
       f. After training: save model as mountaincar_dqn_no_wrapper.pth

5. TASK 4 — PPO ON MOUNTAINCARCONTINUOUS:
       a. Create MountainCarContinuous-v0 wrapped in Monitor(env, log_dir)
       b. Instantiate PPO("MlpPolicy", ...) with standard hyperparameters
       c. model.learn(total_timesteps=200_000)
       d. Save model as ppo_mountaincar_continuous.zip
       e. Plot training curve: load monitor.csv via ts2xy; smooth with 50-ep rolling avg
       f. Evaluate with model.predict(obs, deterministic=True) for 10 episodes
       g. Plot evaluation rewards; compute mean and std
       h. Run human-render test episode; print total reward
```

---

## How to Run

### Open in Google Colab

The notebook is designed for Google Colab with CPU execution. No GPU is required.

1. Upload `Sparse_Reward_RL_DQN_PPO_Curriculum_Learning_MountainCar.ipynb` to Google Drive or GitHub.
2. Open with Google Colab.
3. Run all cells sequentially (**Runtime → Run all**).

### Install Dependencies

The first cell installs all required runtime packages:

```bash
pip install gymnasium[classic-control] pyvirtualdisplay stable-baselines3
apt-get install -y xvfb python-opengl ffmpeg
```

For a local environment:

```bash
pip install torch gymnasium[classic-control] stable-baselines3 matplotlib numpy pyvirtualdisplay
```

### Estimated Execution Time

| Section | Estimated Time |
|---------|---------------|
| Task 1 — Vanilla DQN (150 episodes) | < 1 min |
| Task 2 — Reward Shaping (400 episodes) | 2–4 min |
| Task 3 — Curriculum DQN (1000 episodes) | 5–10 min |
| Task 4 — PPO (200 000 steps) | 3–8 min |
| **Total** | **~10–25 min** |

### Modifying Task Parameters

```python
# Task 1 — change episode budget or network size
EPISODES_MAX = 150      # increase to see if vanilla DQN ever succeeds
n_hidden = 12           # increase for more expressive Q-network

# Task 2 — tune energy shaping
k_energy = 90.0         # increase for stronger energy-based signal
goal_bonus = 50.0       # increase/decrease terminal goal reward
EPISODES = 400          # increase if convergence is slower

# Task 3 — tune curriculum breadth and exploration
EPISODES = 1000
EPSILON_DECAY = 0.995   # slower decay → more exploration throughout
hidden_dim = 64         # network capacity

# Task 4 — tune PPO
total_timesteps = 200_000   # increase for harder continuous environments
ent_coef = 0.01             # increase for more exploration
clip_range = 0.2            # lower for more conservative updates
```

---

## Results

### Task 1 — Vanilla DQN

The agent consistently receives a total reward of **−200 per episode** (always reaching the 200-step timeout), confirming that the standard DQN cannot solve `MountainCar-v0` with the default sparse reward. The loss curve fluctuates without consistent decrease, and the success rate is **0%** across all 150 episodes. This establishes the baseline failure case and motivates the subsequent methods.

### Task 2 — Energy-Based Reward Shaping

| Metric | Outcome |
|--------|---------|
| Training episodes | 400 |
| Convergence | Agent begins reaching the flag within ≈ 150–200 episodes |
| Shaped reward trend | Clear upward trajectory; running average becomes positive |
| Original reward | Still negative (−1/step), but episodes shorten as the agent succeeds |

The energy-based shaping provides dense feedback at every step, allowing the DQN to learn a momentum-building strategy. However, large reward spikes from the energy bonus introduce variance that is visible in the reward curve.

### Task 3 — Reverse Curriculum Learning

| Metric | Outcome |
|--------|---------|
| Training episodes | 1 000 |
| Final success rate (100-ep rolling) | Near-perfect (≥ 80–90%) |
| Reward variance | Lower than Task 2; smoother curve |
| Loss curve | Steady decrease and stabilisation |

The curriculum agent learns the original sparse reward signal without any manual potential engineering. The 100-episode rolling success rate chart demonstrates a clear, monotonic increase toward near-perfect performance.

### Task 4 — PPO on MountainCar Continuous

| Metric | Outcome |
|--------|---------|
| Training timesteps | 200 000 |
| Learning curve | Smooth, monotonic increase from ≈ −20 to ≈ 0 |
| Evaluation reward | Tightly clustered around 0 across all 10 episodes |
| Test episode total reward | ≈ −0.00017 (effectively optimal) |

PPO successfully solves `MountainCarContinuous-v0` within 200 000 timesteps. The entropy coefficient `ent_coef=0.01` provides sufficient exploration to avoid the sparse reward trap without destabilising policy updates.

---

## Analysis and Discussion

### Why Vanilla DQN Fails on Sparse Rewards

The root cause is a **bootstrap failure**: DQN's Bellman update propagates value estimates backward from terminal states. If the agent never reaches the goal by random exploration, the Q-values for all states and actions remain uniformly low (reflecting only the −1 per step signal), and ε-greedy policy improvement has no gradient toward success. In a 200-step episode with a random policy, the probability of reaching the goal flag is extremely small, making this failure mode almost deterministic for the standard DQN without augmentation.

### Reward Shaping vs. Curriculum Learning

Both methods address the same root cause — absence of gradient signal toward the goal — but through different mechanisms:

Reward shaping **injects information about the goal** into the reward function via the energy potential. This accelerates early learning by making every step informative but introduces artificial bias (the shaped reward does not equal the true task reward) and can cause policy miscalibration if the shaping constant `k_energy` is poorly tuned.

Curriculum learning **restructures the experience distribution** so that the agent always has meaningful success feedback to learn from, without altering the reward function. This preserves the integrity of the true objective and produces more stable, lower-variance training, but requires a mechanism to schedule the starting state distribution.

The empirical conclusion: reward shaping achieves faster early convergence (within 400 episodes) but higher variance, while the curriculum approach achieves more robust performance (near-perfect success rate) at the cost of longer training (1000 episodes). For deployment-critical applications, curriculum learning is preferred; for rapid prototyping, reward shaping is a faster option.

### PPO vs. Alternative Methods for Continuous Control

PPO was chosen over alternative continuous-control algorithms for the following reasons:

| Algorithm | Pros | Cons | Decision |
|-----------|------|------|---------|
| PPO | On-policy, stable, minimal hyperparameter tuning, native SB3 support | Less sample-efficient than off-policy | **Selected** |
| TD3 | Off-policy, sample-efficient | Requires target policy smoothing, delayed updates — extra hyperparameters | Not used |
| SAC | Off-policy, entropy regularisation for exploration | Entropy temperature tuning, more complex | Not used |
| TRPO | Theoretical optimality guarantees | No native SB3 support, slow per-update computation | Not used |

For a benchmark as comparatively simple as `MountainCarContinuous-v0`, the added complexity of TD3 or SAC does not yield a measurable performance benefit over PPO. PPO's `ent_coef=0.01` provides sufficient exploration to overcome the sparse reward without requiring manual noise injection (as in TD3's target policy smoothing) or entropy temperature tuning (as in SAC).

### Energy-Based Potential and the Potential-Based Shaping Theorem

The choice of total mechanical energy as the shaping potential is principled: it satisfies the **potential-based shaping condition** $F(s, s') = \gamma \Phi(s') - \Phi(s)$ when $\gamma = 1$ (undiscounted case) or approximately so for $\gamma$ close to 1. Under this condition, Ng et al. (1999) prove that the optimal policy under the shaped reward is identical to the optimal policy under the original reward — the shaping does not corrupt the task objective. The large `k_energy=90.0` scale factor ensures the energy signal dominates the constant −1 step penalty, providing meaningful directional guidance throughout the valley.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `Python` | ≥ 3.8 | Runtime environment |
| `torch` | ≥ 1.13 | DQN network, autograd, Adam optimiser |
| `numpy` | ≥ 1.21 | Array operations, reward logging, moving averages |
| `gymnasium` | ≥ 0.26 | `MountainCar-v0` and `MountainCarContinuous-v0` environments |
| `stable-baselines3` | ≥ 1.7 | PPO implementation, Monitor wrapper, VecNormalize |
| `matplotlib` | ≥ 3.4 | Training curves, reward/loss visualisation |
| `pyvirtualdisplay` | ≥ 3.0 | Virtual framebuffer for Colab rendering |
| `xvfb` (apt) | — | X virtual framebuffer backend |
| `ffmpeg` (apt) | — | Video encoding for episode rendering |

Install all Python dependencies locally:

```bash
pip install torch gymnasium[classic-control] stable-baselines3 matplotlib numpy pyvirtualdisplay
```

> System packages (`xvfb`, `ffmpeg`) are installed via `apt-get` in the notebook's first cell and are pre-available in the Google Colab environment.

---

## Notes and Limitations

- **Task 1 intentional failure:** The vanilla DQN is not expected to solve MountainCar. Its failure is the pedagogical objective of Task 1, establishing the baseline that motivates Tasks 2–4.
- **Stochastic results:** Tasks 1–3 do not fix the random seed for environment resets or mini-batch sampling. Episode rewards, convergence speed, and success rates will vary across runs. The qualitative conclusions (shaping helps; curriculum outperforms shaping on stability) are consistent, but specific numerical results may differ.
- **Task 3 is labelled "curriculum-free" in the code:** The function `train_dqn_curriculum_free()` trains without the full staged curriculum wrapper but implements progressively widened initialisation. The name reflects the absence of an explicit curriculum scheduler class, not the absence of curriculum-based logic.
- **PPO evaluation environment mismatch:** The `evaluate_ppo()` function loads a `VecNormalize` wrapper from `ppo_vecnorm.pkl`. This file is only generated if observation normalisation was applied during training. If `VecNormalize` was not used, the evaluation cell must be modified to use the raw `gym.make` environment directly.
- **Render mode incompatibility in Colab:** `render_mode="human"` in the test episode (Task 4) requires the virtual display (`pyvirtualdisplay`) to be active. If the display fails to start, replace with `render_mode="rgb_array"` and use `plt.imshow` to view frames.
- **Memory management for long runs:** The replay buffer in Tasks 2–3 has `memory_limit=50000`. On Colab's default 12 GB RAM, this is well within bounds. For significantly longer training runs or larger state representations, monitor RAM usage via Colab's runtime panel.
- **Stable Baselines 3 version compatibility:** The `ts2xy` import path (`stable_baselines3.common.results_plotter`) and `VecNormalize.load` API are stable from SB3 ≥ 1.7. Earlier versions may have different import paths.

---

## Author

**Umer Ahmed Baig Mughal** <br>
Master's in Robotics and Artificial Intelligence <br>
*Specialization: Machine Learning · Computer Vision · Human-Robot Interaction · Autonomous Systems · Robotic Motion Control*

---

## License

This project is intended for **academic and research use**. It was developed as part of the *Machine Learning for Robotics* course within the MSc Robotics and Artificial Intelligence program at ITMO University. Redistribution, modification, and use in derivative academic work are permitted with appropriate attribution to the original author.

---

*Lab 5 — Machine Learning for Robotics | MSc Robotics and Artificial Intelligence | ITMO University*

