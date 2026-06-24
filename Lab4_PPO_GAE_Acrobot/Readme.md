# Lab 4 — Advanced Policy Gradient: PPO with GAE on Acrobot-v1

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.26%2B-green)
![Field](https://img.shields.io/badge/Field-Advanced%20Policy%20Gradient-purple)
![License](https://img.shields.io/badge/License-Academic-yellow)

> **Course:** Machine Learning for Robotics — Faculty of Control Systems and Robotics, ITMO University <br>
> **Author:** Umer Ahmed Baig Mughal — MSc Robotics and Artificial Intelligence <br>
> **Topic:** PPO · Clipped Surrogate Objective · GAE · Actor-Critic · Separate Policy and Value Networks · Mini-Batch Updates · Gradient Clipping · Acrobot-v1

---

## Table of Contents

1. [Objective](#objective)
2. [Theoretical Background](#theoretical-background)
   - [From REINFORCE to Actor-Critic](#from-reinforce-to-actor-critic)
   - [Generalised Advantage Estimation (GAE)](#generalised-advantage-estimation-gae)
   - [Proximal Policy Optimisation (PPO)](#proximal-policy-optimisation-ppo)
   - [Clipped Surrogate Objective](#clipped-surrogate-objective)
   - [Mini-Batch Epoch Updates](#mini-batch-epoch-updates)
   - [Gradient Clipping](#gradient-clipping)
   - [Entropy Regularisation](#entropy-regularisation)
   - [System Properties](#system-properties)
3. [Tasks and Implementation](#tasks-and-implementation)
   - [Network Architecture — PolicyNet and ValueNet](#network-architecture--policynet-and-valuenet)
   - [GAE Computation](#gae-computation)
   - [PPO Update Step](#ppo-update-step)
   - [Training Loop](#training-loop)
   - [Inference and Model Checkpoint](#inference-and-model-checkpoint)
4. [Validation and Metrics](#validation-and-metrics)
5. [System Parameters](#system-parameters)
6. [Implementation](#implementation)
   - [File Structure](#file-structure)
   - [Function and Class Reference](#function-and-class-reference)
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

This lab implements Proximal Policy Optimisation (PPO) — the state-of-the-art on-policy deep reinforcement learning algorithm — applied to the Acrobot-v1 underactuated robotics task from the Gymnasium classic control suite. The lab advances beyond the basic policy gradient methods of Lab 3 by introducing an Actor-Critic architecture with separate policy and value networks, Generalised Advantage Estimation (GAE) for low-variance advantage computation, the PPO clipped surrogate objective for safe large-batch policy updates, and multi-epoch mini-batch optimisation over collected rollouts.

The key learning outcomes are:

- Implementing separate `PolicyNet` and `ValueNet` MLP architectures with independent Adam optimisers, following the standard Actor-Critic decomposition that decouples the policy improvement signal from the value baseline estimation.
- Implementing the GAE algorithm (`compute_gae`) using a single reverse-pass accumulation over a collected multi-episode batch, computing per-step advantages `A_t` and discounted returns `G_t` simultaneously from the TD residuals and a λ-weighted exponential moving sum.
- Implementing the full PPO update (`ppo_update`) with the clipped surrogate objective — computing importance sampling ratios `r_t(θ) = π_θ(a|s) / π_θ_old(a|s)`, applying the `clip(r, 1−ε, 1+ε)` constraint, and taking the element-wise minimum of clipped and unclipped surrogates to form a pessimistic policy improvement bound.
- Performing `POLICY_UPDATES = 6` full passes over each collected rollout batch, shuffling and sub-sampling into `MINI_BATCH_SIZE = 32` mini-batches at each pass, reusing collected experience for multiple gradient steps in a sample-efficient manner that would cause divergence in standard policy gradient but is stabilised by the PPO clip.
- Clipping gradient norms of both policy and value networks to `MAX_GRAD_NORM = 0.5` to prevent parameter updates from exploding when batch variance is high.
- Training on the Acrobot-v1 environment — a two-link underactuated pendulum requiring precise torque application — and achieving stable swingup in under 170 episodes with fixed random seed `SEED = 45`.
- Saving and loading full model checkpoints (policy + value network state dicts) to `ppo_acrobot.pth` and running 10 greedy inference episodes to confirm that the learnt policy generalises stably to held-out seeds.

The lab is implemented as a single Jupyter notebook (`PPO_GAE_Actor_Critic_Acrobot.ipynb`), structured around the classical two-loop training pipeline (outer episode loop, inner step loop) with a batch collection phase and a batch update phase that fires every `TRAIN_BATCH_EPISODES = 6` episodes.

---

## Theoretical Background

### From REINFORCE to Actor-Critic

The REINFORCE algorithm (Lab 3) uses the total episode return `G` as the weight for each policy gradient step. This is unbiased but has very high variance — the return `G` is a single Monte Carlo sample of the true expected return, and its variance scales with episode length. Actor-Critic methods reduce this variance by replacing `G` with a lower-variance advantage estimate `A(s, a)`:

```
∇_θ J(θ) ≈ E_τ [ Σ_t ∇_θ log π(a_t|s_t; θ) · A(s_t, a_t) ]

where:
    A(s, a) = Q(s, a) − V(s)        advantage: how much better action a is vs. the average
    V(s)                             value baseline estimated by the Critic (ValueNet)
```

The Actor (PolicyNet) learns the policy `π(a|s; θ)`. The Critic (ValueNet) learns the state value function `V(s; φ)`. Their gradients are computed independently with separate optimisers, which is more stable than sharing network parameters.

### Generalised Advantage Estimation (GAE)

GAE (Schulman et al., 2016) provides a unified framework for advantage estimation that interpolates between two extremes via a parameter λ ∈ [0, 1]:

```
δ_t = r_t + γ · V(s_{t+1}) · mask_t − V(s_t)        (TD residual / 1-step advantage)

A_t^{GAE(γ,λ)} = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}      (exponentially-weighted sum of future TD residuals)
```

Special cases:
- **λ = 0:** `A_t = δ_t = r_t + γV(s_{t+1}) − V(s_t)` — pure 1-step TD advantage. Low variance, high bias (relies on `V` approximation accuracy).
- **λ = 1:** `A_t = Σ r_t − V(s_0)` — Monte Carlo advantage. Zero bias, high variance (long trajectories).
- **λ = 0.95 (used here):** Practical balance — captures multi-step credit assignment while remaining lower-variance than full Monte Carlo.

Computed efficiently via a single reverse pass over the collected batch:

```python
def compute_gae(rewards, masks, values, next_value, gamma, lam):
    values = np.append(values, next_value)        # bootstrap terminal value
    gae = 0.0
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step+1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        advantages[step] = gae
        returns[step] = gae + values[step]         # G_t = A_t + V(s_t)
    return advantages, returns
```

The `mask` variable is 0.0 for terminal states (zeroing out the bootstrap from `V(s')`) and 1.0 otherwise — this correctly handles episode boundaries within a multi-episode batch.

### Proximal Policy Optimisation (PPO)

Standard policy gradient methods take a single gradient step per batch of collected data, then discard it. This is sample-inefficient. Performing multiple gradient steps on the same batch would be more efficient, but large policy updates from multi-step optimisation can move the policy far from the data collection policy, violating the on-policy assumption and causing catastrophic performance collapse.

PPO solves this by constraining how much the policy can change per update via the clipped surrogate objective — allowing multiple optimisation steps per batch without diverging.

The key quantity is the probability ratio:

```
r_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t) = exp(log_prob_new − log_prob_old)
```

`r_t = 1` means the new and old policies assign the same probability to `a_t`. `r_t > 1` means the new policy makes `a_t` more likely than the data-collection policy did.

### Clipped Surrogate Objective

PPO constrains the ratio `r_t` by taking the pessimistic minimum of two objectives:

```
L_CLIP(θ) = E_t [ min( r_t(θ) · A_t,   clip(r_t(θ), 1−ε, 1+ε) · A_t ) ]

where:
    ε = clip_eps = 0.3                (clip range)
    surr1 = r_t · A_t                 (unclipped: full gradient if ratio is small)
    surr2 = clip(r_t, 1−ε, 1+ε) · A_t (clipped: ratio capped at [0.7, 1.3])
    policy_loss = −min(surr1, surr2)   (negated for gradient ascent via loss.backward())
```

The clip prevents `r_t` from moving too far from 1.0 in either direction:
- If `A_t > 0` (good action): clipping at `1+ε` prevents over-reinforcing the action beyond `ε`.
- If `A_t < 0` (bad action): clipping at `1−ε` prevents over-suppressing the action beyond `ε`.

The `min` selects the more conservative (lower) estimate at each step, making the objective a lower bound on the true policy improvement — a safe, pessimistic policy update.

```python
ratio  = torch.exp(mb_new_log_probs - mb_old_log_probs)
surr1  = ratio * mb_advantages
surr2  = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_advantages
policy_loss = -torch.min(surr1, surr2).mean() - ENTROPY_COEF * entropy
```

### Mini-Batch Epoch Updates

After collecting `TRAIN_BATCH_EPISODES = 6` episodes, PPO performs `POLICY_UPDATES = 6` full passes (epochs) over the collected data, sub-sampling shuffled mini-batches of size `MINI_BATCH_SIZE = 32` at each pass:

```
for epoch in range(POLICY_UPDATES):             # 6 full passes
    shuffle(indices)
    for start in range(0, dataset_size, MINI_BATCH_SIZE):   # mini-batches of 32
        mb = batch[start : start + MINI_BATCH_SIZE]
        # compute PPO loss on mini-batch, gradient step
```

This reuses each collected transition up to `POLICY_UPDATES` times, dramatically improving sample efficiency over single-step methods. The PPO clip prevents the policy from drifting too far from `π_θ_old` across these multiple epochs, keeping the updates safe.

### Gradient Clipping

Both policy and value network gradients are clipped by global norm before each parameter update:

```python
nn.utils.clip_grad_norm_(policy_net.parameters(), MAX_GRAD_NORM)    # clip at 0.5
nn.utils.clip_grad_norm_(value_net.parameters(),  MAX_GRAD_NORM)    # clip at 0.5
```

Global norm clipping rescales the entire parameter gradient vector to have L2 norm ≤ `MAX_GRAD_NORM = 0.5` if it exceeds the threshold, without changing the gradient direction. This prevents explosive updates when batch variance is high early in training, and is especially important for the value network which receives dense MSE regression loss signals.

### Entropy Regularisation

An entropy bonus is added to the policy loss to prevent premature collapse to a deterministic policy:

```
L_total = L_CLIP + VALUE_COEF · L_value − ENTROPY_COEF · H(π)

where:
    H(π) = −Σ_a π(a|s) log π(a|s)    (policy entropy)
    ENTROPY_COEF = 0.02               (entropy weight)
    VALUE_COEF   = 0.7                (value loss weight)
```

The `VALUE_COEF = 0.7` scales the value regression loss relative to the policy loss, which use different magnitude gradients. The entropy coefficient of 0.02 is slightly higher than the 0.01 used in Lab 3's REINFORCE, reflecting Acrobot's longer-horizon exploration requirements.

### System Properties

| Property | Value | Notes |
|---|---|---|
| Environment | Acrobot-v1 | Underactuated 2-link pendulum |
| State space | 6-dimensional continuous | `[cos θ₁, sin θ₁, cos θ₂, sin θ₂, θ̇₁, θ̇₂]` |
| Action space | 3 discrete | Torque: −1, 0, +1 N·m on joint 2 |
| Episode termination | Tip height > threshold | `−cos(θ₁) − cos(θ₁+θ₂) > 1.0` |
| Reward per step | −1 (until goal) | Sparse: minimise steps to reach goal |
| Optimal episode length | ~80–100 steps | Fewer steps = higher total reward |
| Success criterion | Reward > −100 | Stable swingup within ~100 steps |
| PolicyNet architecture | `[6 → 64 → 64 → 3]` | ReLU, outputs logits for Categorical |
| ValueNet architecture | `[6 → 64 → 64 → 64 → 1]` | ReLU, 3 hidden layers, scalar output |
| Optimiser (policy) | Adam | lr = 3e-4 |
| Optimiser (value) | Adam | lr = 1e-3 (10× faster than policy) |
| Device | CUDA if available, else CPU | Auto-detected at runtime |
| Random seed | 45 | Applied to torch, numpy, random, and env resets |

---

## Tasks and Implementation

### Network Architecture — PolicyNet and ValueNet

**Goal:** Define separate Actor and Critic networks with independent parameter sets and optimisers.

```python
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=64):
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, action_dim)          # raw logits — no softmax
        )
    def forward(self, x):
        return self.net(x)                         # logits fed to Categorical(logits=...)

class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden=64):
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),  # extra hidden layer vs PolicyNet
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)             # scalar value per state
```

ValueNet has one additional hidden layer (3 hidden vs. 2 in PolicyNet). This asymmetry is intentional: value function estimation is a harder regression problem than policy output (which only needs to distinguish 3 actions), so the deeper Critic benefits from additional representational capacity.

---

### GAE Computation

**Goal:** Compute per-step advantages `A_t` and discounted returns `G_t` from a collected multi-episode batch using the GAE formula.

```python
def compute_gae(rewards, masks, values, next_value, gamma=0.99, lam=0.95):
    values = np.append(values, next_value)       # append bootstrap value at end
    gae = 0.0
    returns     = np.zeros_like(values)
    advantages  = np.zeros_like(values[:-1])

    for step in reversed(range(len(rewards))):
        delta          = rewards[step] + gamma * values[step+1] * masks[step] - values[step]
        gae            = delta + gamma * lam * masks[step] * gae
        advantages[step] = gae
        returns[step]    = gae + values[step]    # G_t = A_t^GAE + V(s_t)

    return advantages, returns[:-1]
```

The `mask` correctly handles within-batch episode boundaries: when `done=True`, `mask=0.0` zeroes both the bootstrap value `γV(s')` and the running GAE accumulation, restarting the advantage computation cleanly at the next episode's first step.

---

### PPO Update Step

**Goal:** Perform `POLICY_UPDATES` epochs of mini-batch gradient updates on the collected rollout using the clipped surrogate objective and MSE value regression.

```python
def ppo_update(policy_net, value_net, optim_policy, optim_value, batch, clip_eps=0.3):
    # Normalise advantages across the full batch before mini-batch splits
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(POLICY_UPDATES):              # 6 epochs
        np.random.shuffle(idxs)
        for start in range(0, dataset_size, MINI_BATCH_SIZE):    # mini-batches of 32
            # --- Policy loss ---
            ratio   = torch.exp(new_log_probs - old_log_probs)
            surr1   = ratio * mb_advantages
            surr2   = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean() - ENTROPY_COEF * entropy

            optim_policy.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(policy_net.parameters(), MAX_GRAD_NORM)
            optim_policy.step()

            # --- Value loss ---
            value_preds = value_net(mb_obs)
            value_loss  = VALUE_COEF * (mb_returns - value_preds).pow(2).mean()

            optim_value.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(value_net.parameters(), MAX_GRAD_NORM)
            optim_value.step()
```

Note that policy and value losses are computed and backpropagated in separate `.backward()` calls with separate optimisers. This prevents gradient interference between the policy improvement signal and the value regression signal — a key design choice distinguishing this implementation from shared-network variants where both losses are summed before a single backward pass.

---

### Training Loop

**Goal:** Collect rollouts across `TRAIN_BATCH_EPISODES = 6` episodes, compute GAE, call `ppo_update`, and track 100-episode rolling average reward.

```python
for episode in range(1, MAX_EPISODES+1):           # outer: episode loop
    obs, _ = env.reset(seed=SEED + episode)        # deterministic per-episode seed
    for step in range(MAX_STEPS):                  # inner: step loop
        # sample action from policy, step env, collect transition
        batch_obs/actions/log_probs/rewards/masks/values.append(...)
        if done: break

    ep_count_for_batch += 1
    if ep_count_for_batch >= TRAIN_BATCH_EPISODES: # batch trigger
        next_value = value_net(last_obs)           # bootstrap value
        advantages, returns = compute_gae(...)
        ppo_update(policy_net, value_net, ..., batch)
        # clear all batch buffers, reset ep_count_for_batch = 0

    if episode % PRINT_INTERVAL == 0:
        print rolling average reward
```

The classical two-loop structure (outer episode, inner step) is preserved as required. The batch accumulation and update phases are clearly separated from the rollout collection phase, making the data flow explicit.

---

### Inference and Model Checkpoint

**Goal:** Save trained policy and value networks, reload from checkpoint, and evaluate for 10 episodes.

```python
# Save
torch.save({
    'policy_state_dict': policy_net.state_dict(),
    'value_state_dict':  value_net.state_dict(),
}, SAVE_PATH)                                     # "ppo_acrobot.pth"

# Load
ckpt = torch.load(model_path, map_location=DEVICE)
policy_net.load_state_dict(ckpt['policy_state_dict'])
value_net.load_state_dict(ckpt['value_state_dict'])
policy_net.eval()
value_net.eval()

# Inference: stochastic policy sampling (not pure greedy)
with torch.no_grad():
    dist   = Categorical(logits=policy_net(obs_tensor))
    action = dist.sample().item()
```

Inference uses stochastic policy sampling rather than greedy `argmax`. For a well-trained policy on a discrete action space with sharp probability distributions, this is equivalent to greedy in expectation, but more robust to borderline cases where two actions have similar probabilities.

---

## Validation and Metrics

### Episode Reward

Acrobot's reward is `−1` per step until the terminal condition (tip above threshold) is met. Total episode reward is therefore:

```
total_reward = −(number of steps to solve)
```

| Metric | Formula | Target |
|---|---|---|
| Episode reward | `−T` where T = steps to goal | > −100 (solve in < 100 steps) |
| Rolling100 average | `mean(last 100 episode rewards)` | Converging toward −100 |
| Inference reward | 10-episode mean at `eval()` mode | Stable > −100 |

A perfect agent solves Acrobot in ~80 steps (reward ≈ −80). An untrained random agent typically takes 400–500 steps (reward ≈ −500).

### Rolling Average Convergence

The training loop maintains a `deque(maxlen=100)` rolling reward window and prints `Rolling100` every `PRINT_INTERVAL = 10` episodes. The key convergence signal is the rolling average rising steadily from ~−500 (random) toward −100 and stabilising there.

### Loss Monitoring

Policy loss and value loss are computed at each mini-batch update. While not explicitly logged to a list in this implementation, convergence is indicated by:
- Policy loss decreasing and stabilising near zero (ratio `r_t ≈ 1` → clipping inactive → loss ≈ entropy bonus).
- Value loss decreasing as `V(s)` learns to approximate `G_t` accurately.

---

## System Parameters

### Environment Parameters

| Parameter | Value | Description |
|---|---|---|
| `ENV_NAME` | `"Acrobot-v1"` | Gymnasium environment ID |
| `SEED` | 45 | Global seed for reproducibility |
| State dim | 6 | `cos θ₁, sin θ₁, cos θ₂, sin θ₂, θ̇₁, θ̇₂` |
| Action dim | 3 | Discrete torques: −1, 0, +1 N·m |

### Training Parameters

| Parameter | Value | Description |
|---|---|---|
| `MAX_EPISODES` | 170 | Total training episode budget |
| `MAX_STEPS` | 500 | Maximum steps per episode |
| `TRAIN_BATCH_EPISODES` | 6 | Episodes collected before each PPO update |
| `gamma` | 0.99 | Discount factor |
| `lam` | 0.95 | GAE λ — advantage smoothing parameter |
| `POLICY_UPDATES` | 6 | Gradient epochs per collected batch |
| `MINI_BATCH_SIZE` | 32 | Mini-batch size within each epoch |
| `clip_eps` | 0.3 | PPO probability ratio clip range `[0.7, 1.3]` |
| `MAX_GRAD_NORM` | 0.5 | Global gradient norm clip threshold |
| `PRINT_INTERVAL` | 10 | Episode interval for console logging |

### Network and Optimiser Parameters

| Parameter | Value | Description |
|---|---|---|
| PolicyNet hidden | 64 | Width of 2 hidden layers |
| ValueNet hidden | 64 | Width of 3 hidden layers |
| `LR_POLICY` | 3e-4 | Adam learning rate for PolicyNet |
| `LR_VALUE` | 1e-3 | Adam learning rate for ValueNet (3.3× higher) |
| `ENTROPY_COEF` | 0.02 | Entropy bonus weight in policy loss |
| `VALUE_COEF` | 0.7 | Value loss scaling factor |

### Checkpoint

| Parameter | Value | Description |
|---|---|---|
| `SAVE_PATH` | `ppo_acrobot.pth` | PyTorch checkpoint (policy + value state dicts) |
| Inference episodes | 10 | Held-out evaluation rollouts after training |
| Inference seed offset | `SEED + 100 + ep` | Ensures inference seeds differ from training seeds |

---

## Implementation

### File Structure

```
Lab4/
├── Readme.md
├── PPO_GAE_Actor_Critic_Acrobot.ipynb    # Complete PPO implementation
└── ppo_acrobot.pth                                      # Saved model checkpoint (generated at runtime)
```

No external data files are required. The Acrobot environment is provided by Gymnasium and initialised in-notebook.

---

### Function and Class Reference

#### `PolicyNet(obs_dim, action_dim, hidden)` — Actor network

Two-hidden-layer MLP outputting raw action logits for a `Categorical` distribution.

| Argument | Type | Default | Description |
|---|---|---|---|
| `obs_dim` | int | — | Observation dimension (6 for Acrobot) |
| `action_dim` | int | — | Number of discrete actions (3 for Acrobot) |
| `hidden` | int | 64 | Width of each hidden layer |

Returns: `logits` — Tensor `(B, action_dim)` of unnormalised log-probabilities.

---

#### `ValueNet(obs_dim, hidden)` — Critic network

Three-hidden-layer MLP outputting a scalar state value estimate.

| Argument | Type | Default | Description |
|---|---|---|---|
| `obs_dim` | int | — | Observation dimension (6 for Acrobot) |
| `hidden` | int | 64 | Width of each hidden layer |

Returns: `value` — Tensor `(B,)` of scalar state value estimates.

---

#### `compute_gae(rewards, masks, values, next_value, gamma, lam)` — GAE utility

Computes per-step advantages and discounted returns for a multi-episode batch via reverse-pass TD-λ accumulation.

| Argument | Type | Default | Description |
|---|---|---|---|
| `rewards` | np.ndarray `(T,)` | — | Collected step rewards |
| `masks` | np.ndarray `(T,)` | — | 1.0 = non-terminal, 0.0 = terminal step |
| `values` | np.ndarray `(T,)` | — | Value estimates at each collected state |
| `next_value` | float | — | Bootstrap value at the last collected state |
| `gamma` | float | 0.99 | Discount factor |
| `lam` | float | 0.95 | GAE λ smoothing parameter |

Returns: `(advantages, returns)` — both `np.ndarray (T,)`.

---

#### `ppo_update(policy_net, value_net, optim_policy, optim_value, batch, clip_eps)` — PPO gradient update

Performs `POLICY_UPDATES` epochs of shuffled mini-batch gradient steps using the clipped surrogate objective and MSE value regression with separate backward passes.

| Argument | Type | Default | Description |
|---|---|---|---|
| `policy_net` | `PolicyNet` | — | Online Actor network |
| `value_net` | `ValueNet` | — | Online Critic network |
| `optim_policy` | `Adam` | — | Policy optimiser |
| `optim_value` | `Adam` | — | Value optimiser |
| `batch` | dict | — | Keys: `obs, actions, old_log_probs, returns, advantages` |
| `clip_eps` | float | 0.3 | PPO clip range |

---

#### `train()` — main training loop

Runs the full training pipeline: episode/step data collection, GAE computation on every 6-episode batch, PPO update, rolling reward tracking, and model checkpoint save.

Returns: `(SAVE_PATH, episode_rewards, rolling_avg)` — checkpoint path and plotting data.

---

#### `inference(model_path, episodes, render)` — evaluation function

Loads checkpoint, sets networks to `eval()` mode, and runs `episodes` stochastic rollouts, printing per-episode reward.

| Argument | Type | Default | Description |
|---|---|---|---|
| `model_path` | str | — | Path to `.pth` checkpoint |
| `episodes` | int | 10 | Number of evaluation rollouts |
| `render` | bool | False | Enable `render_mode="human"` if True |

---

### Algorithm Walkthrough

Complete pipeline (`PPO_GAE_Actor_Critic_Acrobot.ipynb`):

```
1. SETUP:
       Set ENV_NAME="Acrobot-v1", SEED=45, DEVICE=cuda/cpu
       torch.manual_seed(45), np.random.seed(45), random.seed(45)
       Define all hyperparameters

2. NETWORK INSTANTIATION:
       policy_net = PolicyNet(6, 3, hidden=64).to(DEVICE)
       value_net  = ValueNet(6, hidden=64).to(DEVICE)          ← 3 hidden layers
       optim_policy = Adam(policy_net.parameters(), lr=3e-4)
       optim_value  = Adam(value_net.parameters(), lr=1e-3)

3. TRAINING — train():
       Initialise batch buffers, reward tracking deques

       OUTER LOOP: for episode in range(1, 171):
           env.reset(seed=SEED + episode)                       ← deterministic per-episode seed

           INNER LOOP: for step in range(500):
               obs_tensor → PolicyNet → Categorical → sample action
               obs_tensor → ValueNet  → scalar value
               env.step(action) → next_obs, reward, done
               Append (obs, action, log_prob, reward, mask, value) to batch buffers
               if done: break

           Append ep_reward to episode_rewards, reward_deque, rolling_avg
           ep_count_for_batch += 1

           if ep_count_for_batch >= 6:                         ← BATCH UPDATE TRIGGER
               Bootstrap: next_value = ValueNet(last_obs)
               advantages, returns = compute_gae(rewards, masks, values, next_value)
               Build batch dict as torch tensors
               ppo_update(policy_net, value_net, ..., batch):
                   Normalise advantages (zero mean, unit std)
                   for epoch in range(6):                       ← 6 PPO epochs
                       shuffle indices
                       for mini-batch in chunks of 32:
                           Compute ratio = exp(new_log_probs − old_log_probs)
                           surr1 = ratio × advantages
                           surr2 = clip(ratio, 0.7, 1.3) × advantages
                           policy_loss = −min(surr1, surr2) − 0.02·entropy
                           policy_loss.backward() → clip grad norm (0.5) → optim_policy.step()
                           value_loss = 0.7 · MSE(returns, value_preds)
                           value_loss.backward() → clip grad norm (0.5) → optim_value.step()
               Clear all batch buffers, ep_count_for_batch = 0

           Every 10 episodes: print avg reward + Rolling100

       torch.save(policy + value state_dicts, "ppo_acrobot.pth")
       Return (SAVE_PATH, episode_rewards, rolling_avg)

4. EVALUATION — inference():
       Load checkpoint → policy_net.eval(), value_net.eval()
       for ep in range(10):
           Stochastic rollout (no argmax, use dist.sample())
           Print per-episode reward

5. PLOT:
       plt.plot(episode_rewards, label="Episode Reward")
       plt.plot(rolling_avg, label="Rolling Avg")
       Title: "PPO on Acrobot-v1"
```

---

## How to Run

### Open in Google Colab or Jupyter

The notebook runs on CPU or GPU. CUDA is auto-detected and used if available.

1. Upload `PPO_GAE_Actor_Critic_Acrobot.ipynb` to Google Colab or a local Jupyter environment.
2. Install dependencies (first run only):

```bash
pip install gymnasium[classic-control] torch numpy matplotlib
```

3. Run all cells sequentially. The `__main__` cell executes `train()` then `inference()` automatically.
4. After training, `ppo_acrobot.pth` is saved in the working directory (`/content/` on Colab).

### Install Dependencies (Local)

```bash
pip install torch gymnasium matplotlib numpy
```

No virtual display or `xvfb` is needed unless you set `render=True` in the `inference()` call. Headless inference (default) runs without any display server.

### Estimated Execution Time

| Section | CPU Time | GPU Time |
|---|---|---|
| Training — 170 episodes | 5–12 min | 2–4 min |
| Inference — 10 episodes | < 30 sec | < 30 sec |
| Plotting | < 5 sec | < 5 sec |
| **Total** | **~6–13 min** | **~3–5 min** |

### Modifying Hyperparameters

```python
# Extend training if not converging
MAX_EPISODES = 300          # more episodes → more training time

# Widen the policy update trust region
clip_eps = 0.3              # increase for larger per-step updates (riskier)
                            # decrease for more conservative updates (slower)

# Control sample reuse
POLICY_UPDATES = 6          # more epochs per batch → better sample efficiency
                            # too many → policy diverges from old_log_probs

# Tune advantage smoothing
lam = 0.95                  # lower → less variance, more bias (closer to TD)
                            # higher → more variance, less bias (closer to MC)

# Tune exploration
ENTROPY_COEF = 0.02         # increase if policy collapses prematurely
                            # decrease once policy is sufficiently exploratory

# Tune learning rate balance
LR_POLICY = 3e-4            # lower if policy oscillates
LR_VALUE  = 1e-3            # value should converge faster than policy
```

---

## Results

### Training Convergence

With `SEED = 45` and the tuned hyperparameters, the PPO agent on Acrobot-v1 shows the following progression:

| Episode Range | Typical Reward | Description |
|---|---|---|
| 1–30 | ~−500 | Random-like behaviour; policy still mostly uniform |
| 30–80 | −300 to −200 | Policy beginning to find partial swing trajectories |
| 80–130 | −200 to −100 | Consistent partial swingup; GAE estimates improving |
| 130–170 | ~−100 or better | Stable swingup achieved; rolling average converged |

### Inference Results

After 170 episodes of training, the `inference()` function runs 10 episodes with stochastic policy sampling on seeds `SEED + 100` through `SEED + 109`. A well-trained agent consistently achieves rewards above −100, demonstrating that the PPO policy has generalised to unseen initial conditions beyond the training seed distribution.

### Key Advantages Over Lab 3 Baselines

| Aspect | DQN (Lab 3) | REINFORCE+RTG (Lab 3) | PPO (Lab 4) |
|---|---|---|---|
| Algorithm family | Value-based | Policy-based | Advanced policy-based (Actor-Critic) |
| Advantage estimation | None (Q-values) | RTG | GAE (λ=0.95) |
| Sample reuse | High (replay buffer) | None | High (6 epochs per batch) |
| Policy stability | Target network | None | Clipped surrogate |
| Environment | CartPole (4D state) | CartPole (4D state) | Acrobot (6D state, 3 actions) |
| Convergence speed | Fast | Slow | Moderate — efficient on harder task |

---

## Analysis and Discussion

### Why PPO Is Preferred Over Vanilla PG for Acrobot

Acrobot's sparse reward structure (−1 every step until goal, up to 500 steps) means that early in training, all trajectories look equally bad — every episode has reward ≈ −500. Vanilla REINFORCE normalises these returns and computes policy gradients that are nearly zero in expectation, leading to extremely slow learning. PPO's Critic (ValueNet) learns to differentiate state quality even before the policy succeeds, providing informative advantage signals `A_t = r_t + γV(s_{t+1}) − V(s_t)` that reflect incremental progress toward the goal even when the total episode return is still −500.

### The Role of GAE λ = 0.95

With `lam = 0.95`, GAE computes advantages as an exponentially-weighted sum of TD residuals that extends roughly `1 / (1 − 0.95) = 20` steps into the future on average. This is well-matched to Acrobot's structure: the last ~20 steps before the goal are the most informative for the policy (maximum torque in the right direction), and the GAE window captures exactly this credit assignment horizon. A pure 1-step advantage (`lam = 0`) would fail to propagate credit back to the sequence of preparatory torque applications needed to build swing momentum.

### Separate Optimisers vs. Combined Loss

Using separate `optim_policy` and `optim_value` with independent `.backward()` calls prevents the value regression gradient from contaminating the policy gradient direction (and vice versa). When a shared loss `L = L_policy + w · L_value` is used with a single backward pass, the value gradient can inadvertently pull the policy network toward value-minimising parameter regions that may not improve the policy. The separate-optimiser approach is more stable at the cost of two backward passes per mini-batch.

### Deterministic Per-Episode Seeds

Each training episode is reset with `seed = SEED + episode`, making every episode's initial state deterministic given the seed. This reduces the variance in the reward curve compared to purely random resets, and ensures that the rolling average reflects genuine policy improvement rather than lucky or unlucky initial configurations. The inference seeds are offset by 100 to ensure the evaluation uses genuinely unseen configurations not encountered during training.

### clip_eps = 0.3 vs. Canonical 0.2

The canonical PPO clip value is 0.2. Using `clip_eps = 0.3` here allows larger policy updates per epoch — the ratio can move up to 30% away from 1.0 rather than 20%. This wider trust region accelerates early training when the policy is far from optimal, at the cost of slightly higher variance in the updates. For Acrobot's 170-episode training budget, the wider clip trades some stability for faster convergence, which is appropriate given the tight episode budget.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| Python | ≥ 3.8 | Runtime environment |
| torch | ≥ 2.0 | Neural networks, autograd, distributions, model save/load |
| gymnasium | ≥ 0.26 | Acrobot-v1 environment |
| numpy | ≥ 1.21 | Array operations, GAE computation, batch processing |
| matplotlib | ≥ 3.4 | Episode reward and rolling average plots |
| collections | stdlib | `deque(maxlen=100)` for rolling reward window |
| math, random, time | stdlib | Utilities, seeding, timing |

Install all third-party dependencies:

```bash
pip install torch gymnasium matplotlib numpy
```

All packages are available in Google Colab without manual installation.

---

## Notes and Limitations

- **`train()` references `SAVE_PATH` as a global:** The `SAVE_PATH = "ppo_acrobot.pth"` assignment is in the `__main__` block and accessed as a global inside `train()`. In a modular codebase, `SAVE_PATH` should be passed as a function argument to avoid implicit global dependency.
- **Inference uses stochastic sampling, not greedy:** `dist.sample()` is used in `inference()` rather than `dist.probs.argmax()`. For a sharp, near-deterministic policy this makes no practical difference, but for reporting purposes a deterministic greedy evaluation (ε = 0) is more standard for benchmarking.
- **Rolling100 deque vs. true 100-episode window:** `reward_deque = deque(maxlen=100)` correctly implements a 100-episode rolling window. Early in training (episodes 1–99) the mean is over fewer than 100 samples, so the reported `Rolling100` is actually a shorter-window average until episode 100. This is cosmetic but worth noting when interpreting early convergence plots.
- **No learning rate scheduler:** Unlike Lab 3's `StepLR` on the policy gradient, this implementation uses fixed Adam learning rates for both networks. A cosine annealing or linear decay schedule could further stabilise late-stage training once the policy has largely converged.
- **Mini-batch remainder handling:** `range(0, dataset_size, MINI_BATCH_SIZE)` naturally handles the last mini-batch being smaller than 32 (if `dataset_size % 32 ≠ 0`) without explicit padding or dropping. PyTorch's loss `.mean()` correctly normalises by the actual mini-batch size.
- **GPU memory:** With `MINI_BATCH_SIZE = 32` and `MAX_EPISODES × MAX_STEPS` batch sizes at most a few thousand transitions, GPU memory usage is negligible. The `map_location=DEVICE` in `torch.load` ensures checkpoints saved on GPU can be loaded on CPU and vice versa.
- **Classical two-loop structure preserved:** Per the lab requirement, the training pipeline explicitly maintains the outer episode loop and inner step loop. The batch accumulation and PPO update logic is kept separate from the rollout collection logic within this structure, preserving readability of the data flow.

---

## Author

**Umer Ahmed Baig Mughal**
Master's in Robotics and Artificial Intelligence <br>
Specialization: Machine Learning · Computer Vision · Human-Robot Interaction · Autonomous Systems · Robotic Motion Control

---

## License

This project is intended for academic and research use. It was developed as part of the Machine Learning for Robotics course within the MSc Robotics and Artificial Intelligence program at ITMO University. Redistribution, modification, and use in derivative academic work are permitted with appropriate attribution to the original author.

---

*Lab 4 — Machine Learning for Robotics | MSc Robotics and Artificial Intelligence | ITMO University*

