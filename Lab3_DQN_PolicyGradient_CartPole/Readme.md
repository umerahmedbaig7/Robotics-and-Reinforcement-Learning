# Lab 3 — Deep Reinforcement Learning: DQN, Target Networks & Policy Gradient on CartPole

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.26%2B-green)
![Field](https://img.shields.io/badge/Field-Deep%20Reinforcement%20Learning-purple)
![License](https://img.shields.io/badge/License-Academic-yellow)

> **Course:** Machine Learning for Robotics — Faculty of Control Systems and Robotics, ITMO University <br>
> **Author:** Umer Ahmed Baig Mughal — MSc Robotics and Artificial Intelligence <br>
> **Topic:** DQN · Experience Replay · Target Network · ε-Greedy Exploration · Policy Gradient · Reward-To-Go · Entropy Regularisation · CartPole-v1

---

## Table of Contents

1. [Objective](#objective)
2. [Theoretical Background](#theoretical-background)
   - [Deep Q-Network (DQN)](#deep-q-network-dqn)
   - [Experience Replay](#experience-replay)
   - [Target Network](#target-network)
   - [ε-Greedy Exploration](#ε-greedy-exploration)
   - [Simple Policy Gradient (REINFORCE)](#simple-policy-gradient-reinforce)
   - [Reward-To-Go](#reward-to-go)
   - [Entropy Regularisation](#entropy-regularisation)
   - [System Properties](#system-properties)
3. [Tasks and Implementation](#tasks-and-implementation)
   - [Part 1 — DQN with Experience Replay](#part-1--dqn-with-experience-replay)
   - [Part 1 — DQN with Target Network](#part-1--dqn-with-target-network)
   - [Part 2 — Simple Policy Gradient (REINFORCE)](#part-2--simple-policy-gradient-reinforce)
   - [Part 2 — Reward-To-Go Policy Gradient](#part-2--reward-to-go-policy-gradient)
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

This lab implements and compares two fundamental families of deep reinforcement learning algorithms — value-based (DQN) and policy-based (Policy Gradient) — applied to the CartPole-v1 control task from the Gymnasium classic control suite. The lab progresses from a baseline DQN with experience replay through stabilised training via a target network, then transitions to direct policy optimisation using REINFORCE and its improved variant with Reward-To-Go.

The key learning outcomes are:

- Implementing a Deep Q-Network (DQN) with a configurable MLP architecture and training it via the Bellman optimality equation on mini-batches sampled from a circular experience replay buffer, achieving stable learning in under 1000 episodes on CartPole-v0.
- Tuning the four key DQN hyperparameters — ε-decay schedule, hidden layer count and width, learning rate, and replay/update frequency — to achieve episode rewards consistently above the 195-step threshold.
- Extending the base DQN class with a Target Network (`DQNTarget`) that maintains a periodically-synced frozen copy of the online network for computing bootstrap targets, eliminating the moving-target instability inherent in vanilla DQN.
- Implementing the REINFORCE (Simple Policy Gradient) algorithm with a stochastic Categorical policy, episode-level batch updates, and an ε-greedy warm-up exploration schedule that decays into pure policy sampling.
- Adding an entropy bonus to the policy gradient loss to prevent premature policy collapse and maintain exploration throughout training.
- Integrating a StepLR learning rate scheduler to prevent gradient oscillations during late-stage policy gradient training.
- Replacing the total episode return used as policy weights with Reward-To-Go — a causally correct, lower-variance return estimate computed via a reverse-pass discounted sum — and observing the resulting improvement in learning stability.
- Evaluating all four trained agents (DQN, DQN+Target, PG, PG+RTG) on 10 greedy inference episodes to confirm generalisation to unseen environment rollouts.

The lab is implemented as a single Google Colab Jupyter notebook (`Deep_RL_DQN_Policy_Gradient_CartPole.ipynb`) executed on CPU, producing live training loss plots, episode reward curves with a success threshold line, and printed inference rewards.

---

## Theoretical Background

### Deep Q-Network (DQN)

DQN approximates the action-value function `Q*(s, a)` — the expected cumulative discounted reward for taking action `a` in state `s` and following the optimal policy thereafter — using a neural network parameterised by `θ`:

```
Q(s, a; θ) ≈ Q*(s, a)
```

The network is trained by minimising the mean-squared Bellman error over mini-batches of transitions `(s, a, r, s', done)` sampled from the replay buffer:

```
L(θ) = MSE(Q(s, a; θ),  y)

where:
    y = r + γ · max_{a'} Q(s', a'; θ)  · (1 − done)      (Bellman target)
    γ = discount factor
```

Only the Q-value at the taken action index `a` is updated in the target vector; all other action entries retain their current predicted values. This prevents spurious gradient updates on actions that were not taken.

### Experience Replay

Vanilla online Q-learning updates the network after each transition `(s, a, r, s')`. This creates two problems: consecutive transitions are highly correlated (breaking the i.i.d. assumption of SGD), and rare but informative transitions are seen only once. Experience replay addresses both by:

1. Storing all observed transitions in a fixed-capacity circular buffer (deque): `memory`.
2. At each update step, sampling a uniformly random mini-batch of `batch_size` transitions.
3. Computing the Bellman targets and updating the network on this decorrelated batch.

```
memory.append((state, action, reward, next_state, done))
if len(memory) > memory_max:
    memory.pop(0)                        # FIFO eviction — oldest transition removed

batch = random.sample(memory, batch_size)
```

The replay buffer acts as a data-efficiency mechanism, allowing each transition to contribute to multiple gradient updates and smoothing the loss landscape during training.

### Target Network

In standard DQN, both the current Q-values and the Bellman targets are computed from the same network `θ`. As `θ` updates, the targets shift — a feedback loop known as the moving-target problem that destabilises training and can cause divergence.

The Target Network (`θ⁻`) is a periodically-frozen copy of the online network used exclusively for computing bootstrap targets:

```
y = r + γ · max_{a'} Q(s', a'; θ⁻) · (1 − done)        (target from frozen network)

Every target_update steps:
    θ⁻ ← θ                                               (hard sync)
```

The online network `θ` is updated every step; the target network `θ⁻` is updated only every `target_update` steps. This decouples the target from the rapidly changing online weights, providing a stable regression objective and dramatically reducing oscillation and divergence.

```python
class DQNTarget(DQN):
    def update(self, states, targets):
        super().update(states, targets)
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())   # hard sync
```

### ε-Greedy Exploration

The ε-greedy policy balances exploration (trying random actions to discover better strategies) and exploitation (selecting the greedy action according to current Q-values):

```
π_ε(s) = { random action         with probability ε
          { argmax_a Q(s, a; θ)  with probability 1 − ε
```

`ε` is initialised to 1.0 (fully random) and decayed multiplicatively each episode:

```
ε ← max(ε_min, ε · ε_decay)
```

A slower decay (e.g. 0.995 per episode) maintains exploration longer, helping the agent discover rewarding trajectories before committing to a policy. The minimum `ε_min = 0.01` ensures a small residual exploration probability is always maintained.

### Simple Policy Gradient (REINFORCE)

Policy Gradient methods directly parameterise and optimise the policy `π(a|s; θ)` without maintaining a Q-value table. The REINFORCE algorithm uses the policy gradient theorem to compute an unbiased gradient estimate of the expected return:

```
∇_θ J(θ) = E_τ [ Σ_t ∇_θ log π(a_t|s_t; θ) · G ]

where:
    G = Σ_t r_t                    total episode return (undiscounted sum)
    π(a|s; θ)                      stochastic Categorical policy over actions
    log π(a_t|s_t; θ)              log-probability of the taken action
```

The loss used in practice (negated for gradient ascent via `loss.backward()`):

```
L_PG = −mean( log_prob(a_t) · ŵ_t )

where:
    ŵ_t = (G − mean(G)) / (std(G) + ε)      normalised return weights
```

Weight normalisation (zero-mean, unit-variance) prevents large returns from causing gradient explosion and stabilises early training.

### Reward-To-Go

The standard REINFORCE weight `G` assigns the total episode return equally to every action taken in the episode — including actions that occurred *after* the reward was received. This violates causality: an action at time `t` cannot affect rewards received before `t`.

Reward-To-Go replaces `G` with the causally correct future return from each timestep:

```
G_t = Σ_{t'=t}^{T} γ^{t'-t} · r_{t'}        (discounted sum from t onward)
```

Implemented via a single reverse-pass accumulation:

```python
def compute_rewards_to_go(rewards, gamma=0.99):
    n = len(rewards)
    rtg = np.zeros(n)
    future_reward = 0
    for i in reversed(range(n)):
        future_reward = rewards[i] + gamma * future_reward
        rtg[i] = future_reward
    return rtg
```

RTG reduces variance in the gradient estimate compared to using the total return `G`, leading to faster convergence and more stable learning. Actions early in the episode receive larger weights (more future reward attributable), and actions near termination receive smaller weights.

### Entropy Regularisation

To prevent premature convergence to a deterministic policy before the agent has adequately explored the state space, an entropy bonus is added to the policy gradient loss:

```
L_total = L_PG − β · H(π)

where:
    H(π) = −Σ_a π(a|s) log π(a|s)    policy entropy (higher = more uniform = more exploration)
    β = entropy_beta                   entropy weight (0.01)
```

A high-entropy policy assigns more uniform probabilities across actions, encouraging exploration. As training progresses and the agent finds good strategies, the policy naturally sharpens while the entropy term prevents it from collapsing too early.

### System Properties

| Property | Value | Notes |
|---|---|---|
| Environment (Part 1 baseline) | CartPole-v0 | 200-step episode limit |
| Environment (Part 1 target / Part 2) | CartPole-v1 | 500-step episode limit |
| State space | 4-dimensional continuous | `[cart_pos, cart_vel, pole_angle, pole_ang_vel]` |
| Action space | 2 discrete | Push left (0) or push right (1) |
| Success threshold (v0) | 195 | Average reward over 100 consecutive episodes |
| Success threshold (v1) | 495 | Average reward over 100 consecutive episodes |
| DQN architecture | `[4 → 64 → 64 → n_actions]` | ReLU activations |
| PG architecture | `[4 → 128 → 128 → 2]` (base) / `[4 → 64 → 64 → 2]` (RTG) | ReLU activations |
| DQN optimiser | Adam | lr = 0.003 (base), 0.005 (target) |
| PG optimiser | Adam + StepLR | lr = 0.01, step_size=200, γ=0.9 |
| Hardware | CPU (Google Colab) | No GPU required |

---

## Tasks and Implementation

### Part 1 — DQN with Experience Replay

**Goal:** Train a DQN with circular experience replay on CartPole-v0 and achieve stable rewards above 195 within 250 episodes.

```python
class DQN():
    def __init__(self, state_dim, action_dim, hidden_dim=64, alpha=0.003):
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)
```

**Bellman target computation:**

```python
q_next  = model.model(torch.Tensor(next_states_b)).detach().numpy()
targets = model.model(torch.Tensor(states_b)).detach().numpy()
for i in range(batch_size):
    targets[i, actions_b[i]] = rewards_b[i] + gamma * np.max(q_next[i]) * (not dones_b[i])
model.update(states_b, targets)
```

The `.detach()` call prevents gradient flow through the target computation — the Bellman target is treated as a fixed label, not a moving part of the computation graph.

**Key tuned hyperparameters:**

```python
epsilon_decay  = 0.995     # slower decay → more exploration
gamma          = 0.99      # high discount → long-horizon planning
hidden_dim     = 64        # 2 hidden layers of 64 units
batch_size     = 32
update_freq    = 4         # update network every 4 steps
memory_max     = 5000
```

---

### Part 1 — DQN with Target Network

**Goal:** Stabilise DQN training on CartPole-v1 by decoupling the regression target from the online network using a periodically-synced frozen copy.

```python
class DQNTarget(DQN):
    def __init__(self, state_dim, action_dim, hidden_dim=64, alpha=0.005, target_update=10):
        super().__init__(state_dim, action_dim, hidden_dim, alpha)
        self.target_model = copy.deepcopy(self.model)     # frozen copy
        self.target_update = target_update
        self.steps = 0

    def update(self, states, targets):
        super().update(states, targets)
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())   # hard sync

    def predict_target(self, states):
        with torch.no_grad():
            return self.target_model(torch.Tensor(states))
```

**Training uses target network for bootstrap targets:**

```python
q_next  = model.predict_target(next_states_b).numpy()   # from frozen θ⁻
targets = model.model(torch.Tensor(states_b)).detach().numpy()
for i in range(batch_size):
    targets[i, actions_b[i]] = rewards_b[i] + gamma * np.max(q_next[i]) * (not dones_b[i])
model.update(states_b, targets)                          # updates online θ only
```

---

### Part 2 — Simple Policy Gradient (REINFORCE)

**Goal:** Train a stochastic policy network on CartPole-v1 using the REINFORCE algorithm with ε-greedy warm-up, entropy regularisation, and batch episode updates.

```python
class PG():
    def get_policy(self, obs):
        logits = self.model(torch.FloatTensor(obs))
        return distributions.Categorical(logits=logits)   # stochastic action distribution

    def update(self, batch_states, batch_actions, batch_weights):
        weights = (weights - weights.mean()) / (weights.std() + 1e-8)   # normalise
        log_probs = policy_dist.log_prob(actions)
        entropy   = policy_dist.entropy().mean()
        loss      = -torch.mean(log_probs * weights) - self.entropy_beta * entropy
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()                             # decay lr every 200 updates
```

**Episode-level batch collection and update:**

```python
# Collect full episode
G = sum(episode_rewards)                              # total undiscounted return
batch_weights.extend([G] * len(episode_rewards))      # same weight for all steps

# Update every BATCH_SIZE=10 episodes
if (i_episode + 1) % BATCH_SIZE == 0:
    model_pg.update(batch_states, batch_actions, batch_weights)
    batch_states, batch_actions, batch_weights = [], [], []
```

---

### Part 2 — Reward-To-Go Policy Gradient

**Goal:** Replace the total episode return `G` with causally correct per-step Reward-To-Go weights and observe the reduction in gradient variance.

```python
def compute_rewards_to_go(rewards, gamma=0.99):
    n = len(rewards)
    rtg = np.zeros(n)
    future_reward = 0
    for i in reversed(range(n)):
        future_reward = rewards[i] + gamma * future_reward
        rtg[i] = future_reward
    return rtg
```

**RTG replaces flat `G` as the policy weight:**

```python
rtg = compute_rewards_to_go(episode_rewards, gamma=0.99)
batch_weights.extend(rtg)                             # per-step discounted future return

# Update every BATCH_EPISODES=15 episodes
if (i_episode + 1) % BATCH_EPISODES == 0:
    model_pg.update(batch_states, batch_actions, batch_weights)
```

The RTG variant uses a smaller architecture `[64, 64]` (vs `[128, 128]` for plain PG) and updates every 15 episodes instead of 10, reflecting that lower-variance gradients allow less frequent but more stable updates.

---

## Validation and Metrics

### DQN — Episode Reward

Training success is assessed against the CartPole-v1 benchmark threshold:

| Metric | Formula | Target |
|---|---|---|
| Episode reward | `Σ_t r_t` over one episode | ≥ 195 (v0), ≥ 495 (v1) |
| Inference reward | Greedy rollout, ε = 0 | Single-episode score at convergence |

The reward curve is plotted every 50 episodes with the threshold line at 195 overlaid.

### DQN — Training Loss

The MSE Bellman loss `L(θ) = MSE(Q(s,a;θ), y)` is logged at every gradient step and plotted in real time. A decreasing and smoothing loss curve indicates the network is converging toward a consistent Q-function.

### Policy Gradient — Episode Reward

PG performance is assessed by the raw episode reward curve. The goal line is set at 495 (CartPole-v1 maximum). Stable rewards above 495 for extended episode windows indicate a solved policy.

### Inference Evaluation

All four agents are evaluated greedily (ε = 0) on 10 held-out rollouts after training:

```python
for ep in range(10):
    while not done:
        action = model.get_action(state, epsilon=0.0)    # PG greedy
        # or: action = argmax Q(s, a; θ)                 # DQN greedy
```

The 10-episode mean and individual scores confirm whether learning has generalised or was episodically lucky.

---

## System Parameters

### DQN — Base (CartPole-v0)

| Parameter | Value | Description |
|---|---|---|
| `EPISODES_MAX` | 250 | Training episode budget |
| `STEPS_MAX` | 200 | Maximum steps per episode |
| `epsilon` (init) | 1.0 | Initial exploration rate |
| `epsilon_min` | 0.01 | Minimum exploration rate |
| `epsilon_decay` | 0.995 | Multiplicative decay per episode |
| `gamma` | 0.99 | Discount factor |
| `alpha` | 0.003 | Adam learning rate |
| `hidden_dim` | 64 | Width of both hidden layers |
| `batch_size` | 32 | Replay mini-batch size |
| `update_freq` | 4 | Steps between gradient updates |
| `memory_max` | 5000 | Circular replay buffer capacity |

### DQN — Target Network (CartPole-v1)

| Parameter | Value | Description |
|---|---|---|
| `EPISODES_MAX` | 500 | Training episode budget |
| `STEPS_MAX` | 200 | Maximum steps per episode |
| `epsilon_decay` | 0.99 | Faster decay than base DQN |
| `gamma` | 0.95 | Slightly lower discount |
| `alpha` | 0.005 | Higher learning rate |
| `hidden_dim` | 64 | Width of both hidden layers |
| `batch_size` | 16 | Smaller replay mini-batch |
| `update_freq` | 12 | Less frequent gradient updates |
| `memory_max` | 5000 | Circular replay buffer capacity |
| `target_update` | 10 | Steps between target network hard syncs |

### Policy Gradient (REINFORCE)

| Parameter | Value | Description |
|---|---|---|
| `EPISODES_MAX` | 1000 | Training episode budget |
| `STEPS_MAX` | 500 | Maximum steps per episode |
| `hidden_layers` | `[128, 128]` | Two wider hidden layers |
| `alpha` | 0.01 | Adam learning rate |
| `entropy_beta` | 0.01 | Entropy bonus coefficient |
| `BATCH_SIZE` | 10 | Episodes per policy update |
| `EPS_START` | 1.0 | Initial ε for warm-up exploration |
| `EPS_END` | 0.01 | Minimum ε |
| `EPS_DECAY` | 0.995 | Multiplicative ε decay per episode |
| LR scheduler | StepLR | step_size=200, γ=0.9 |

### Reward-To-Go Policy Gradient

| Parameter | Value | Description |
|---|---|---|
| `EPISODES_MAX` | 1000 | Training episode budget |
| `STEPS_MAX` | 500 | Maximum steps per episode |
| `hidden_layers` | `[64, 64]` | Smaller architecture than plain PG |
| `alpha` | 0.01 | Adam learning rate |
| `entropy_beta` | 0.01 | Entropy bonus coefficient |
| `BATCH_EPISODES` | 15 | Episodes per policy update (vs 10 for plain PG) |
| `gamma` | 0.99 | Discount factor for RTG computation |

---

## Implementation

### File Structure

```
Lab3/
├── Readme.md
└── Deep_RL_DQN_Policy_Gradient_CartPole.ipynb    # All 4 implementations
```

No external data files are required. The CartPole environment is provided by Gymnasium and initialised in-notebook.

---

### Class Reference

#### `DQN(state_dim, action_dim, hidden_dim, alpha)` — base Q-network

Encapsulates a two-hidden-layer MLP, MSE loss, and Adam optimiser. Provides `update()` for Bellman regression and `predict()` for greedy action selection.

| Argument | Type | Default | Description |
|---|---|---|---|
| `state_dim` | int | — | Observation space dimension (4 for CartPole) |
| `action_dim` | int | — | Number of discrete actions (2 for CartPole) |
| `hidden_dim` | int | 64 | Width of each hidden layer |
| `alpha` | float | 0.003 | Adam learning rate |

**Methods:**

| Method | Signature | Description |
|---|---|---|
| `update` | `(states, targets)` | One gradient step minimising MSE Bellman error |
| `predict` | `(state)` | Returns Q-values for all actions; no gradient |

---

#### `DQNTarget(state_dim, action_dim, hidden_dim, alpha, target_update)` — DQN with frozen target

Extends `DQN` with a `copy.deepcopy` target network that hard-syncs every `target_update` gradient steps.

| Argument | Type | Default | Description |
|---|---|---|---|
| `target_update` | int | 10 | Gradient steps between hard sync `θ⁻ ← θ` |

**Additional methods:**

| Method | Signature | Description |
|---|---|---|
| `predict_target` | `(states)` | Q-values from frozen `θ⁻`; no gradient |
| `update` | `(states, targets)` | Calls super + increments step counter + syncs when due |

---

#### `PG(state_dim, action_dim, hidden_layers, alpha, entropy_beta)` — policy gradient agent

Parameterises a stochastic Categorical policy, computes log-probabilities and entropy, and updates via normalised REINFORCE with optional ε-greedy warm-up.

| Argument | Type | Default | Description |
|---|---|---|---|
| `hidden_layers` | list | `[128, 128]` | Widths of hidden layers (variable depth) |
| `alpha` | float | 0.01 | Adam learning rate |
| `entropy_beta` | float | 0.01 | Entropy bonus coefficient β |

**Methods:**

| Method | Signature | Description |
|---|---|---|
| `get_policy` | `(obs)` | Returns `Categorical` distribution over actions |
| `get_action` | `(obs, epsilon)` | ε-greedy sample from policy |
| `update` | `(batch_states, batch_actions, batch_weights)` | REINFORCE gradient step with entropy bonus |

---

#### `compute_rewards_to_go(rewards, gamma)` — RTG utility

Computes per-timestep discounted future returns via a single reverse-pass accumulation.

| Argument | Type | Default | Description |
|---|---|---|---|
| `rewards` | list / array | — | Episode reward sequence `[r_0, r_1, ..., r_T]` |
| `gamma` | float | 0.99 | Discount factor |

Returns: `np.ndarray` of shape `(T,)` — per-step discounted future return `G_t`.

---

### Algorithm Walkthrough

Complete pipeline (`Deep_RL_DQN_Policy_Gradient_CartPole.ipynb`):

```
1. ENVIRONMENT SETUP:
       pip install gymnasium[classic-control] pyvirtualdisplay xvfb
       Virtual display initialised for headless rendering

2. DQN — BASE (Part 1):
       a. env = CartPole-v0; n_states=4, n_actions=2
       b. Instantiate DQN(4, 2, hidden_dim=64, alpha=0.003)
       c. For each episode (250 max):
            - ε-greedy action selection
            - Store (s, a, r, s', done) in memory (FIFO, max 5000)
            - Every update_freq=4 steps and memory ≥ batch_size=32:
                 * Sample batch from memory
                 * Compute Bellman targets using ONLINE network
                 * Update network via MSE loss
            - Decay ε × 0.995 per episode
            - Plot live loss and reward curves
       d. Inference: greedy rollout, print total reward

3. DQN — TARGET NETWORK (Part 1):
       a. env = CartPole-v1; n_states=4, n_actions=2
       b. Instantiate DQNTarget(4, 2, hidden_dim=64, alpha=0.005, target_update=10)
       c. For each episode (500 max):
            - ε-greedy action selection (decay=0.99)
            - Store in memory (FIFO, max 5000)
            - Every update_freq=12 steps and memory ≥ batch_size=16:
                 * Sample batch from memory
                 * Compute Bellman targets using TARGET network (θ⁻)
                 * Update online network (θ) via MSE loss
                 * Sync θ⁻ ← θ every target_update=10 gradient steps
            - Decay ε per episode
            - Plot live loss and reward curves
       d. Inference: greedy rollout (ε=0), print total reward

4. POLICY GRADIENT — REINFORCE (Part 2):
       a. env = CartPole-v1
       b. Instantiate PG(4, 2, hidden_layers=[128,128], alpha=0.01, entropy_beta=0.01)
       c. For each episode (1000 max):
            - Collect full episode using ε-greedy get_action()
            - Compute G = sum(episode_rewards)
            - Extend batch with [G, G, ..., G] for all episode steps
            - Decay ε × 0.995 per episode
            - Every BATCH_SIZE=10 episodes:
                 * Normalise batch_weights
                 * Compute REINFORCE loss + entropy bonus
                 * Gradient step; StepLR scheduler step
                 * Clear batch buffers
       d. Plot reward curve vs 495 threshold
       e. Inference: 1 greedy rollout + 10 greedy test episodes

5. REWARD-TO-GO (Part 2):
       a. env = CartPole-v1
       b. Instantiate PG(4, 2, hidden_layers=[64,64], alpha=0.01, entropy_beta=0.01)
       c. For each episode (1000 max):
            - Collect full episode using pure policy get_action()
            - Compute rtg = compute_rewards_to_go(episode_rewards, gamma=0.99)
            - Extend batch with per-step rtg values (not flat G)
            - Every BATCH_EPISODES=15 episodes:
                 * Normalise batch_weights
                 * REINFORCE + entropy gradient step
                 * Clear batch buffers
       d. Plot reward curve vs 495 threshold
       e. Inference: 1 greedy rollout + 10 greedy test episodes
```

---

## How to Run

### Open in Google Colab

The notebook is designed for Google Colab with CPU execution. No GPU is required.

1. Upload `Deep_RL_DQN_Policy_Gradient_CartPole.ipynb` to Google Drive or GitHub.
2. Open with Google Colab.
3. Run the first cell to install dependencies:

```bash
!pip install gymnasium[classic-control] pyvirtualdisplay
!apt-get install -y xvfb python-opengl ffmpeg
```

4. Run all cells sequentially (`Runtime → Run all`).

### Install Dependencies (Local)

```bash
pip install torch gymnasium matplotlib numpy pyvirtualdisplay
```

For headless rendering outside Colab, a virtual display is required:

```bash
sudo apt-get install xvfb
```

### Estimated Execution Time

| Section | Estimated Time |
|---|---|
| DQN base — 250 episodes | 2–4 min |
| DQN Target Network — 500 episodes | 4–8 min |
| Policy Gradient REINFORCE — 1000 episodes | 5–10 min |
| Reward-To-Go PG — 1000 episodes | 5–10 min |
| All inference evaluations | < 1 min |
| **Total** | **~15–35 min** |

### Modifying Hyperparameters

```python
# DQN — tune exploration schedule
epsilon_decay = 0.995    # slower → more exploration; faster → quicker exploitation
epsilon_min   = 0.01     # residual floor

# DQN — tune replay and update frequency
batch_size   = 32        # larger batch → more stable gradients, slower per-step
update_freq  = 4         # lower → more frequent updates; higher → more stable targets
memory_max   = 5000      # larger → more diverse replay; smaller → recency bias

# Target Network — tune sync frequency
target_update = 10       # higher → more stable targets; lower → tracks faster

# Policy Gradient — tune batch size and entropy
BATCH_SIZE    = 10       # more episodes per update → lower variance gradient
entropy_beta  = 0.01     # higher → more exploration; lower → faster policy sharpening

# RTG — tune discount
gamma = 0.99             # lower γ → shorter planning horizon; higher → long-term focus
```

---

## Results

### Part 1 — DQN (CartPole-v0)

The base DQN with experience replay achieves episode rewards above the 195 threshold within approximately 200 episodes. The loss curve shows an initial rise as the Q-function begins fitting non-trivial targets, followed by a gradual decrease as the policy improves and the Bellman errors reduce.

### Part 1 — DQN with Target Network (CartPole-v1)

The Target Network variant produces a noticeably smoother reward curve compared to the base DQN. The hard sync mechanism prevents oscillations caused by the moving-target problem, allowing the agent to achieve stable high rewards on the harder CartPole-v1 environment with a 500-step limit.

| Model | Environment | Episodes to Threshold | Training Stability |
|---|---|---|---|
| DQN (base) | CartPole-v0 (limit 200) | ~200 | Moderate — reward spikes common |
| DQN + Target | CartPole-v1 (limit 500) | ~400–500 | Higher — smoother convergence |

### Part 2 — Policy Gradient vs. Reward-To-Go

| Model | Architecture | Episodes to Stability | Key Advantage |
|---|---|---|---|
| REINFORCE (plain) | `[128, 128]` | ~600–800 | Simple; ε warm-up helps early on |
| REINFORCE + RTG | `[64, 64]` | ~500–700 | Lower variance; more consistent learning |

The RTG variant demonstrates more stable reward growth because each action is credited only with the future return it could causally influence, reducing noise in the gradient estimate. The smaller `[64, 64]` architecture also trains faster per episode, partially compensating for the increased episodes-per-update (15 vs. 10).

### Inference Results

All four agents achieve strong greedy performance after training:

- DQN (base): inference rewards at or near 200 (CartPole-v0 cap)
- DQN + Target: inference rewards at or near 500 (CartPole-v1 cap)
- PG (plain): high inference rewards across 10 test episodes
- PG + RTG: consistent high rewards, lower variance across test episodes

---

## Analysis and Discussion

### Why Experience Replay Stabilises DQN

Without replay, consecutive transitions `(s_t, a_t, r_t, s_{t+1})` are highly correlated — the network receives a stream of nearly identical updates biased toward the current trajectory. Replay breaks this correlation by sampling uniformly from a large buffer, ensuring the network trains on a diverse mix of past experiences. This is the most impactful single stabilisation technique in DQN and was one of the key contributions of the original DeepMind paper.

### The Moving-Target Problem and Its Fix

In standard Q-learning, the target `y = r + γ · max_a Q(s', a; θ)` depends on the same parameters `θ` being optimised. Every gradient step that improves `Q(s, a; θ)` also shifts the target, creating a feedback loop. The Target Network breaks this by computing targets from `θ⁻`, which is frozen for `target_update` steps. This turns the regression into a stable supervised learning problem for short intervals, dramatically reducing divergence. The choice of `target_update` controls the trade-off: too small and `θ⁻` tracks `θ` too closely, losing the stabilisation benefit; too large and `θ⁻` lags too far behind, slowing learning.

### REINFORCE vs. DQN — Fundamental Trade-offs

DQN and Policy Gradient represent two different approaches to the RL objective:

| Aspect | DQN (Value-Based) | REINFORCE (Policy-Based) |
|---|---|---|
| **What is learnt** | Q-function Q(s,a) | Policy π(a\|s) directly |
| **Update frequency** | Every few steps (online) | Every N episodes (batch) |
| **Sample efficiency** | High — replay reuses data | Low — episodes discarded after update |
| **Variance** | Low — Q-function is stable | High — return-weighted gradient is noisy |
| **Convergence** | Stable with target network | Slow without variance reduction |
| **Bias** | Bootstrapping introduces bias | Unbiased gradient estimator |

For CartPole, both approaches are viable but DQN tends to converge faster due to higher sample efficiency. Policy gradient methods become more attractive in continuous action spaces or when the policy structure is important (e.g., for probability outputs in robotics).

### Why Reward-To-Go Reduces Variance

The total return `G` applied as a flat weight to every step in the episode means that early actions are rewarded or penalised for outcomes that happened after them — regardless of whether those outcomes were caused by those actions. RTG `G_t` eliminates this noise by attributing to each action only the return from that point forward. Mathematically, RTG is a control variate that reduces gradient variance without introducing bias, and it is the foundational building block for more advanced variance-reduction techniques like generalised advantage estimation (GAE) used in PPO and A3C.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| Python | ≥ 3.8 | Runtime environment |
| torch | ≥ 2.0 | Neural networks, autograd, distributions |
| gymnasium | ≥ 0.26 | CartPole-v0 / CartPole-v1 environment |
| numpy | ≥ 1.21 | Array operations, batch processing |
| matplotlib | ≥ 3.4 | Live training loss and reward plots |
| pyvirtualdisplay | ≥ 3.0 | Headless virtual display for Colab rendering |
| xvfb / ffmpeg | system | Virtual frame buffer for env rendering |

Install all Python dependencies locally:

```bash
pip install torch gymnasium matplotlib numpy pyvirtualdisplay
```

All Python packages are pre-installed in Google Colab. The `xvfb` and `ffmpeg` system packages are installed via the first notebook cell.

---

## Notes and Limitations

- **CartPole-v0 vs. v1:** The base DQN trains on CartPole-v0 (200-step limit, threshold 195), while all other experiments use CartPole-v1 (500-step limit, threshold 495). This is intentional — v0 is an easier benchmark suitable for the baseline, and v1 is used to test the stabilisation benefit of the Target Network and the higher-horizon challenge for policy gradient methods.
- **FIFO memory with `list.pop(0)`:** The replay buffer uses a Python `list` with `pop(0)` for eviction. This is O(n) per eviction — for `memory_max = 5000` this is negligible, but for larger buffers a `collections.deque(maxlen=N)` would be significantly more efficient.
- **`retain_graph` not needed in DQN:** Unlike the PINN lab (Lab 2), DQN targets are computed with `.detach()` so no second backward pass is needed. The computation graph is freed normally after each `loss.backward()`.
- **PG ε-greedy warm-up is non-standard:** Standard REINFORCE does not use ε-greedy exploration — the stochastic policy provides sufficient exploration naturally. The ε warm-up here helps the agent encounter diverse trajectories early in training before the policy has learnt anything useful, but it biases the gradient estimate since ε-greedy actions do not follow `π(a|s; θ)`. The RTG variant drops ε-greedy entirely, using pure policy sampling.
- **Entropy bonus vs. ε decay coupling:** Both the entropy bonus and ε-decay target exploration, but through different mechanisms. During early training when ε is high, the entropy bonus is somewhat redundant. Its main benefit kicks in after ε has decayed below ~0.1 and the policy gradient is the primary exploration mechanism.
- **Stochastic training results:** Neither DQN nor PG use fixed random seeds in this notebook. Episode reward curves will differ between runs. The qualitative ordering (DQN+Target > DQN, PG+RTG more stable than PG) is consistent, but exact convergence episodes will vary.
- **`show_state` requires `mode='rgb_array'`:** The `env.render(mode='rgb_array')` call in `show_state()` is only compatible with Gymnasium ≤ 0.25 API. In Gymnasium ≥ 0.26, the render mode must be specified at `gym.make('CartPole-v1', render_mode='rgb_array')`. The training loops do not call `show_state()`, so this only affects manual visual inspection cells.

---

## Author

**Umer Ahmed Baig Mughal**
Master's in Robotics and Artificial Intelligence <br>
Specialization: Machine Learning · Computer Vision · Human-Robot Interaction · Autonomous Systems · Robotic Motion Control

---

## License

This project is intended for academic and research use. It was developed as part of the Machine Learning for Robotics course within the MSc Robotics and Artificial Intelligence program at ITMO University. Redistribution, modification, and use in derivative academic work are permitted with appropriate attribution to the original author.

---

*Lab 3 — Machine Learning for Robotics | MSc Robotics and Artificial Intelligence | ITMO University*

