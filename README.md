<div align="center">

# 🤖 Robotics and Reinforcement Learning
### MSc Robotics and Artificial Intelligence — Course Repository

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.26%2B-00C853?style=for-the-badge&logo=openaigym&logoColor=white)](https://gymnasium.farama.org/)
[![StableBaselines3](https://img.shields.io/badge/Stable--Baselines3-Required-7B2FBE?style=for-the-badge)](https://stable-baselines3.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Required-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Required-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)]()
[![Field](https://img.shields.io/badge/Field-Robotics%20%26%20AI-blueviolet?style=for-the-badge&logo=ros&logoColor=white)]()

<br>

> *"Structure matters — whether it comes from a spectrum, an equation, or a replay buffer, the right inductive bias is what makes learning possible."*

<br>

**Author:** Umer Ahmed Baig Mughal <br>
**Programme:** MSc Robotics and Artificial Intelligence <br>
**Specialization:** Machine Learning · Computer Vision · Human-Robot Interaction · Autonomous Systems · Robotic Motion Control <br>
**Institution:** ITMO University — Faculty of Control Systems and Robotics <br>
**Courses:** Robotics & Reinforcement Learning · Machine Learning for Robotics

</div>

---

## 📋 Table of Contents

- [📖 About This Repository](#-about-this-repository)
- [🗂️ Repository Structure](#️-repository-structure)
- [🔬 Course Overview](#-course-overview)
- [🧪 Lab Summaries](#-lab-summaries)
  - [🔵 Lab 1 — SVD Applications: Image Compression, Regression, Eigenfaces, Robust PCA & LASSO](#-lab-1--svd-applications-image-compression-regression-eigenfaces-robust-pca--lasso)
  - [🟠 Lab 2 — Physically Consistent Deep Learning: PINNs, ODE Discovery & Quadrotor Dynamics](#-lab-2--physically-consistent-deep-learning-pinns-ode-discovery--quadrotor-dynamics)
  - [🟣 Lab 3 — Deep Reinforcement Learning: DQN, Target Networks & Policy Gradient on CartPole](#-lab-3--deep-reinforcement-learning-dqn-target-networks--policy-gradient-on-cartpole)
  - [🟢 Lab 4 — Advanced Policy Gradient: PPO with GAE on Acrobot-v1](#-lab-4--advanced-policy-gradient-ppo-with-gae-on-acrobot-v1)
  - [🔴 Lab 5 — Sparse Reward Reinforcement Learning: DQN, Reward Shaping, Curriculum Learning & PPO on MountainCar](#-lab-5--sparse-reward-reinforcement-learning-dqn-reward-shaping-curriculum-learning--ppo-on-mountaincar)
- [🔗 Skill Progression Across Labs](#-skill-progression-across-labs)
- [⚙️ Technical Specifications](#️-technical-specifications)
- [🚀 Getting Started](#-getting-started)
- [📊 Key Results Summary](#-key-results-summary)
- [🧠 Analysis and Discussion](#-analysis-and-discussion)
- [⚠️ Notes and Limitations](#️-notes-and-limitations)
- [🧰 Tech Stack](#-tech-stack)
- [👤 Author](#-author)
- [📄 License](#-license)

---

## 📖 About This Repository

This repository collects five self-contained laboratory assignments completed as part of the MSc Robotics and Artificial Intelligence programme at ITMO University. Each lab is an end-to-end Jupyter notebook (designed for Google Colab, CPU-only) covering a distinct cluster of techniques in data-driven analysis and robot learning. Together they trace a coherent arc from classical matrix decomposition through physics-informed deep learning to advanced policy optimisation on real control tasks.

The unifying thread across all five labs is the idea that **structure matters**: whether that structure comes from the SVD spectrum of a matrix, a governing differential equation, a replay buffer, or a physics prior, imposing the right inductive bias dramatically reduces sample complexity and improves generalisation.

### 🎯 What You Will Find Here

| 📁 Lab | 🏷️ Topic | 🧠 Core Concept | 🛠️ Key Tools |
|:------:|:--------:|:---------------:|:-------------:|
| Lab 1 | SVD Applications | Truncated SVD · Pseudo-Inverse Regression · Eigenfaces · RPCA · LASSO | NumPy · SciPy · scikit-learn |
| Lab 2 | Physically Consistent Deep Learning | PINNs · ODE Discovery · Quadrotor Dynamics | PyTorch (`torch.func`) |
| Lab 3 | Deep RL — DQN & Policy Gradient | Experience Replay · Target Network · REINFORCE · Reward-To-Go | PyTorch · Gymnasium |
| Lab 4 | Advanced Policy Gradient — PPO | GAE · Actor-Critic · Clipped Surrogate · Mini-Batch Updates | PyTorch · Gymnasium |
| Lab 5 | Sparse Reward RL | Reward Shaping · Reverse Curriculum · PPO (SB3) | Gymnasium · Stable-Baselines3 |

---

## 🗂️ Repository Structure

```
📦 Robotics-and-Reinforcement-Learning/
│
├── 📄 README.md                               # ← You are here
│
├── 📁 Lab1_SVD_Applications/
│   ├── 📄 Readme.md
│   ├── 📁 Assets/
│   │   ├── 🖼️ allFaces(1)
│   │   ├── 📊 housing.data
│   │   └── 🖼️ Safer-City-Driving.jpg
│   └── 📓 SVD_Applications_Image_Compression_Regression_Eigenfaces_Robust_PCA_LASSO.ipynb
│
├── 📁 Lab2_PINNs_ODE_Quadrotor/
│   ├── 📄 Readme.md
│   └── 📓 PINNs_ODE_Discovery_Quadrotor_Dynamics.ipynb
│
├── 📁 Lab3_DQN_PolicyGradient_CartPole/
│   ├── 📄 Readme.md
│   └── 📓 Deep_RL_DQN_Policy_Gradient_CartPole.ipynb
│
├── 📁 Lab4_PPO_GAE_Acrobot/
│   ├── 📄 Readme.md
│   ├── 📓 PPO_GAE_Actor_Critic_Acrobot.ipynb
│   └── 🧠 ppo_acrobot.pth                     # generated at runtime
│
└── 📁 Lab5_SparseRewards_MountainCar/
    ├── 📄 Readme.md
    └── 📓 Sparse_Reward_RL_DQN_PPO_Curriculum_Learning_MountainCar.ipynb
```

---

## 🔬 Course Overview

The **Robotics and Reinforcement Learning** / **Machine Learning for Robotics** coursework develops both the theoretical and practical tools for data-driven analysis, physics-informed deep learning, and policy optimisation on real control tasks. The curriculum spans classical matrix decomposition, physically consistent neural network design, and the full spectrum of deep reinforcement learning — from foundational value-based methods to sparse-reward strategies that mirror real deployment challenges.

The five labs progress in a deliberate sequence of increasing complexity, moving steadily closer to real robotics deployment:

**Phase 1 — Structured Data Analysis (Lab 1):** Before any learning-based method is introduced, the Singular Value Decomposition is established as a universal analytical tool — underlying compression, regression, face recognition, robust decomposition, and optimal denoising alike.

**Phase 2 — Physics-Informed Learning (Lab 2):** Neural networks are constrained by known governing equations, progressing from black-box function approximation to joint discovery of unknown physical parameters, culminating in a physically consistent quadrotor dynamics model.

**Phase 3 — Foundational Deep RL (Lab 3):** Value-based (DQN) and policy-based (Policy Gradient) methods are implemented from first principles and compared on the CartPole benchmark.

**Phase 4 — Advanced Policy Optimisation (Lab 4):** PPO with Generalised Advantage Estimation and an Actor-Critic architecture is implemented to solve the harder Acrobot underactuated swingup task.

**Phase 5 — Sparse Reward Strategies (Lab 5):** The MountainCar sparse-reward failure mode is systematically solved through reward shaping, reverse curriculum learning, and PPO — directly relevant to real-world tasks where dense reward signals are unavailable.

```
  Structured Data        Physics-Informed       Foundational RL      Advanced Policy       Sparse Reward
  Analysis                Deep Learning                                Optimisation         Strategies
  ─────────────           ──────────────         ──────────────       ──────────────────    ──────────────
  ┌─────────┐             ┌─────────┐            ┌─────────┐          ┌─────────┐           ┌─────────┐
  │  LAB 1  │──────────►  │  LAB 2  │──────────► │  LAB 3  │────────► │  LAB 4  │─────────► │  LAB 5  │
  └─────────┘             └─────────┘            └─────────┘          └─────────┘           └─────────┘
  SVD / RPCA              PINNs                  DQN + Target Net     PPO + GAE             Reward Shaping
  Eigenfaces              ODE Discovery          REINFORCE / RTG      Actor-Critic          Curriculum
  Optimal Threshold       Quadrotor Dynamics      Experience Replay    Clipped Surrogate     SB3 PPO
```

---

## 🧪 Lab Summaries

---

### 🔵 Lab 1 — SVD Applications: Image Compression, Regression, Eigenfaces, Robust PCA & LASSO

<div align="center">

[![Lab1](https://img.shields.io/badge/Lab%201-SVD%20Applications-0078D7?style=flat-square&logo=python&logoColor=white)]()
[![Course](https://img.shields.io/badge/Course-Data--Driven%20Methods-lightblue?style=flat-square)]()
[![Output](https://img.shields.io/badge/Output-6%20Tasks-lightblue?style=flat-square)]()
[![Notebook](https://img.shields.io/badge/Notebook-DDM__Pract1-lightblue?style=flat-square)]()

</div>

#### 📌 Task Description

> Implement and evaluate a broad set of data-driven analysis techniques, all unified by the **Singular Value Decomposition (SVD)**, across three real datasets: a road photograph, the Boston Housing dataset, and a face image library.

**What the task requires:**
- **Task 1 — Truncated SVD Image Compression:** Implement and compare two rank-selection strategies — a 5%-threshold-based method and a 99%-energy-based method — leaning on the Eckart–Young–Mirsky theorem for optimality guarantees.
- **Task 2 — Linear Regression via SVD Pseudo-Inverse:** Solve the Boston Housing overdetermined system using `x* = Vᵀ Σ⁻¹ Uᵀ b`, with an 80/20 train-test split evaluated via RMSE, MSE, and R².
- **Task 3 — Feature Standardisation and Significance:** Apply column-wise normalisation so regression coefficients become directly interpretable as feature importance scores.
- **Task 4 — Eigenface Reconstruction:** Build an eigenface basis from SVD of a mean-subtracted 36-subject training matrix, then reconstruct a held-out subject at 8 progressive ranks (r = 1 to 1200).
- **Task 5 — Robust PCA via IALM:** Implement the full RPCA algorithm (`shrink()`, `SVT()`, IALM loop) to decompose a face image stack into a low-rank background **L** and sparse shadow/outlier component **S**.
- **Task 6 — Optimal Hard Threshold:** Implement the Gavish–Donoho polynomial `ω(β) = 0.56β³ − 0.95β² + 1.82β + 1.43` and demonstrate correct recovery of true rank-2 structure from a noisy synthetic matrix.

#### 🔑 Key Concepts

| Concept | Description |
|---------|-------------|
| 📐 Truncated SVD | Optimal low-rank approximation per Eckart–Young–Mirsky theorem |
| 📊 Pseudo-Inverse Regression | `x* = Vᵀ Σ⁻¹ Uᵀ b` — solves overdetermined linear systems via SVD |
| 🧮 Feature Standardisation | Normalised coefficients double as importance scores |
| 👤 Eigenfaces | SVD basis of mean-subtracted face matrix; held-out reconstruction at multiple ranks |
| 🧩 Robust PCA (IALM) | `shrink()` + `SVT()` iterative loop separating low-rank **L** from sparse **S** |
| 🎯 Gavish–Donoho Threshold | Polynomial `ω(β)` gives statistically optimal denoising rank, beating ad-hoc thresholds |

#### 📤 Key Numerical Results

```
Datasets: Safer-City-Driving.jpg, housing.data (506×13), allFaces(1) (Yale Face DB B, 192×168)

┌─────────────────────────────┬──────────────────────────────────────────────┐
│ Result                      │ Outcome                                      │
├─────────────────────────────┼──────────────────────────────────────────────┤
│ Truncated SVD (99% energy)  │ Near-lossless image reconstruction at a      │
│                             │ fraction of storage                          │
│ SVD pseudo-inverse (Housing)│ Positive R² on held-out 20% test split       │
│ Eigenface reconstruction    │ Subject 37 reconstructed at r = 1 → 1200     │
│ Gavish–Donoho threshold     │ Correct rank-2 recovery — outperforms ad-hoc │
│                             │ 90%-energy threshold                         │
└─────────────────────────────┴──────────────────────────────────────────────┘
```

**Key result:** All six tasks are instantiations of the SVD — demonstrating that fluency with matrix decomposition is foundational to image processing, regression, face recognition, robust decomposition, sparse recovery, and statistically optimal denoising.

---

### 🟠 Lab 2 — Physically Consistent Deep Learning: PINNs, ODE Discovery & Quadrotor Dynamics

<div align="center">

[![Lab2](https://img.shields.io/badge/Lab%202-PINNs%20%26%20ODE%20Discovery-E85D04?style=flat-square&logo=python&logoColor=white)]()
[![Course](https://img.shields.io/badge/Course-ML%20for%20Robotics-orange?style=flat-square)]()
[![Output](https://img.shields.io/badge/Output-6%20Tasks%20%2B%203%20HW-orange?style=flat-square)]()
[![Notebook](https://img.shields.io/badge/Notebook-MLR__Pract2-orange?style=flat-square)]()

</div>

#### 📌 Task Description

> Implement **Physics-Informed Neural Networks (PINNs)** as a framework for incorporating physical constraints into deep learning models, progressing from vanilla black-box MLPs through ODE discovery with learnable parameters to a physically consistent planar quadrotor model.

**What the task requires:**
- **Task 1 — ODE Solution (Black-Box vs. PINN):** Train a standard MLP on 50 noisy samples from `d²y/dx² + y = 0` in `[-5, 5]`, observe failure outside this range, then add the physics residual loss `L_phys = MSE(d²f/dx², −f)` plus dummy collocation points over `[-10, 10]` to enable accurate extrapolation.
- **HW1 — Periodic Boundary Constraint:** Add a 2π-periodicity loss `MSE(f(−π), f(π)) + MSE(f'(−π), f'(π))` to anchor the solution to the cosine's global structure.
- **Task 2 — ODE Discovery:** Jointly optimise the unknown frequency `ω` in `d²y/dx² + ω²y = 0` with the network weights via `omega_t = torch.tensor(0.75, requires_grad=True)`, recovering the true value ω = 1.5.
- **HW2 / HW3 — Extended Discovery:** Extend ODE discovery to a denser dummy grid, then to a second unknown (bias term B), deriving the third-order operator `d²y/dx² + ω²y − ω²B = 0`.
- **Tasks 3 & 4 — Quadrotor Dynamics:** Train a data-driven MLP on 6-DOF planar quadrotor CSV trajectories to predict state derivatives `[ẏ, ż, φ̇, v̇y, v̇z, ω̇]`, then build a PINN version adding the known analytical equations of motion as a soft physics teacher.

#### 🔑 Key Concepts

| Concept | Description |
|---------|-------------|
| ⚡ Physics Residual Loss | `L_phys = MSE(d²f/dx², −f)` — penalises violation of the governing ODE |
| 🧮 Dummy Collocation Points | Extends physics constraint over full test domain at negligible cost |
| 🔄 Periodicity Constraint | `MSE(f(−π), f(π)) + MSE(f'(−π), f'(π))` — anchors global structure |
| 🔍 ODE Discovery | Learnable parameter (`omega_t`) co-optimised with network weights to recover true physical constants |
| 🚁 Quadrotor PINN | Known 6-DOF equations of motion act as a soft physics teacher for a data-driven model |
| 🧰 `torch.func` | `jacrev`, `hessian`, `vmap` — used for computing PINN derivative losses |

#### 📤 Key Numerical Results

```
Data: Synthetic ODE samples (y = A cos(ωx + φ)); planar quadrotor train.csv / test.csv (gdown)

┌─────────────────────────────┬──────────────────────────────────────────────┐
│ Result                      │ Outcome                                      │
├─────────────────────────────┼──────────────────────────────────────────────┤
│ PINN + dummy collocation    │ Accurate extrapolation to [-10,10] from 50   │
│                             │ noisy points in [-5,5]                       │
│ Joint optimisation (omega_t)│ True ω = 1.5 recovered from data alone       │
│ Quadrotor PINN              │ Matches/outperforms pure data-driven baseline│
│                             │ on all 6 state components                    │
└─────────────────────────────┴──────────────────────────────────────────────┘
```

**Key result:** PINNs outperform black-box models on all ODE tasks, particularly for out-of-distribution extrapolation. The quadrotor PINN matches or exceeds the purely data-driven baseline by leveraging known physics to compensate for sparse and noisy real-world data.

---

### 🟣 Lab 3 — Deep Reinforcement Learning: DQN, Target Networks & Policy Gradient on CartPole

<div align="center">

[![Lab3](https://img.shields.io/badge/Lab%203-DQN%20%26%20Policy%20Gradient-7B2FBE?style=flat-square&logo=python&logoColor=white)]()
[![Course](https://img.shields.io/badge/Course-ML%20for%20Robotics-purple?style=flat-square)]()
[![Env](https://img.shields.io/badge/Env-CartPole--v0%2Fv1-purple?style=flat-square)]()
[![Notebook](https://img.shields.io/badge/Notebook-Lab--3__CartPole-purple?style=flat-square)]()

</div>

#### 📌 Task Description

> Implement and compare two fundamental families of deep reinforcement learning algorithms — **value-based (DQN)** and **policy-based (Policy Gradient)** — applied to the CartPole-v1 control task.

**What the task requires:**
- **DQN with Experience Replay:** Build a circular replay buffer to decouple consecutive transitions, enabling stable Q-learning with a 2-hidden-layer MLP. Tune `ε_decay=0.995`, `batch_size=32`, `update_freq=4`, `memory_max=5000`.
- **DQN with Target Network:** Add a periodically-frozen copy of the online network (`θ⁻`) to compute stable Bellman bootstrap targets, hard-synced (`θ⁻ ← θ`) every `target_update=10` steps.
- **REINFORCE (Simple Policy Gradient):** Train a stochastic `Categorical` policy via the policy gradient theorem with ε-greedy warm-up, batch episode updates, entropy regularisation (`β=0.01`), and a `StepLR` scheduler.
- **Reward-To-Go Policy Gradient:** Implement causally correct per-step return weights `G_t = Σ_{t'≥t} γ^{t'-t} r_{t'}` to reduce gradient variance versus flat total-return weighting.

#### 🔑 Key Concepts

| Concept | Description |
|---------|-------------|
| 🔁 Experience Replay | Circular buffer decouples consecutive transitions for stable Q-learning |
| 🎯 Target Network | Frozen copy `θ⁻`, hard-synced every `target_update=10` steps — eliminates moving-target instability |
| 🎲 REINFORCE | Stochastic `Categorical` policy trained via the policy gradient theorem |
| 📉 Entropy Regularisation | `β=0.01` — encourages exploration, prevents premature policy collapse |
| ⏩ Reward-To-Go | `G_t = Σ_{t'≥t} γ^{t'-t} r_{t'}` — causal return weighting, lower variance than flat returns |

#### 📤 Key Numerical Results

```
Env: CartPole-v0 (DQN baseline) / CartPole-v1 (500-step limit, Target Network + PG)

┌─────────────────────────────┬──────────────────────────────────────────────┐
│ Method                      │ Outcome                                      │
├─────────────────────────────┼──────────────────────────────────────────────┤
│ DQN + Experience Replay     │ Reward ≥ 195 within ~200 episodes            │
│ DQN + Target Network        │ Solves CartPole-v1; smoothest training curve │
│ REINFORCE                   │ Converges with entropy reg. + StepLR         │
│ PG + Reward-To-Go           │ More stable than plain REINFORCE; smaller    │
│                             │ [64,64] net, 15-episode batches              │
└─────────────────────────────┴──────────────────────────────────────────────┘
```

**Key result:** DQN+Target produces the smoothest training curve; PG+RTG is more stable than plain REINFORCE. Both DQN variants and both PG variants achieve near-perfect inference rewards on held-out episodes.

---

### 🟢 Lab 4 — Advanced Policy Gradient: PPO with GAE on Acrobot-v1

<div align="center">

[![Lab4](https://img.shields.io/badge/Lab%204-PPO%20%2B%20GAE-2E7D32?style=flat-square&logo=python&logoColor=white)]()
[![Course](https://img.shields.io/badge/Course-ML%20for%20Robotics-green?style=flat-square)]()
[![Env](https://img.shields.io/badge/Env-Acrobot--v1-green?style=flat-square)]()
[![Notebook](https://img.shields.io/badge/Notebook-LAB4__Advanced__PG-green?style=flat-square)]()

</div>

#### 📌 Task Description

> Implement **Proximal Policy Optimisation (PPO)** with **Generalised Advantage Estimation (GAE)** on the Acrobot-v1 underactuated pendulum task, advancing beyond the basic policy gradient methods of Lab 3.

**What the task requires:**
- **Actor-Critic Architecture:** Build separate `PolicyNet` `[6→64→64→3]` and `ValueNet` `[6→64→64→64→1]` networks with independent Adam optimisers (`lr=3e-4` and `lr=1e-3` respectively).
- **GAE (λ = 0.95):** Implement a single reverse-pass accumulation computing per-step advantages `A_t = Σ_l (γλ)^l δ_{t+l}` and discounted returns `G_t = A_t + V(s_t)`, with episode boundary masking for multi-episode batches.
- **PPO Clipped Surrogate:** Clip the probability ratio `r_t = exp(log_π_new − log_π_old)` to `[1−ε, 1+ε]` (`ε=0.3`) and take the pessimistic minimum of clipped/unclipped surrogates as the policy objective.
- **Mini-Batch Epoch Updates:** Run 6 full passes (`POLICY_UPDATES=6`) over each 6-episode rollout batch, sub-sampled into mini-batches of 32, with global gradient norm clipping at 0.5.
- **Model Checkpoint:** Save policy and value state dicts to `ppo_acrobot.pth`; run 10-episode inference evaluation with stochastic sampling on held-out seeds.

#### 🔑 Key Concepts

| Concept | Description |
|---------|-------------|
| 🎭 Actor-Critic | Asymmetric-depth `PolicyNet`/`ValueNet`, independent optimisers reflect different learning problems |
| 📐 GAE (λ=0.95) | Reverse-pass TD residual accumulation — captures ~20 steps of future credit assignment |
| ✂️ Clipped Surrogate | `ε=0.3` clip on probability ratio — allows safe multi-epoch reuse of rollout data |
| 🧱 Mini-Batch Updates | 6 epochs × 32-sample mini-batches per 6-episode batch; gradient norm clip at 0.5 |
| 💾 Checkpointing | `ppo_acrobot.pth` — policy + value state dicts for reproducible inference |

#### 📤 Key Numerical Results

```
Env: Acrobot-v1, SEED=45, ε_clip=0.3, λ_GAE=0.95

┌─────────────────────────────┬──────────────────────────────────────────────┐
│ Result                      │ Outcome                                      │
├─────────────────────────────┼──────────────────────────────────────────────┤
│ Stable swingup              │ Reward > −100 achieved within 170 episodes   │
│ Generalisation              │ Confirmed across 10 unseen inference seeds   │
└─────────────────────────────┴──────────────────────────────────────────────┘
```

**Key result:** With `SEED=45`, the PPO agent on Acrobot-v1 achieves stable swingup (reward > −100) within 170 episodes. Inference across 10 unseen seeds confirms generalisation well beyond the training seed distribution.

---

### 🔴 Lab 5 — Sparse Reward Reinforcement Learning: DQN, Reward Shaping, Curriculum Learning & PPO on MountainCar

<div align="center">

[![Lab5](https://img.shields.io/badge/Lab%205-Sparse%20Reward%20RL-C62828?style=flat-square&logo=python&logoColor=white)]()
[![Course](https://img.shields.io/badge/Course-ML%20for%20Robotics-red?style=flat-square)]()
[![Env](https://img.shields.io/badge/Env-MountainCar--v0%2FContinuous-red?style=flat-square)]()
[![Notebook](https://img.shields.io/badge/Notebook-LAB__5__MountainCar-red?style=flat-square)]()

</div>

#### 📌 Task Description

> Investigate and solve the **sparse reward problem** using the MountainCar benchmark. Standard DQN cannot solve `MountainCar-v0` because the agent essentially never reaches the goal by random exploration — this lab systematically addresses the failure through three progressively more principled techniques.

**What the task requires:**
- **Task 1 — Vanilla DQN (Baseline Failure):** Confirm a 3-layer DQN with `hidden_dim=12` achieves 0% success across 150 episodes (total reward always −200), establishing the cold-start failure mode.
- **Task 2 — Energy-Based Reward Shaping:** Use total mechanical energy `E(s) = g(pos − pos_min) + 0.5·vel²` as a potential-based shaping function `F(s,s') = k_energy · (E(s') − E(s))` with `k_energy=90.0` and a +50 goal bonus, satisfying the potential-based condition (Ng et al., 1999).
- **Task 3 — Reverse Curriculum Learning:** Start training from states close to the goal, progressively widening the initial state distribution, using the original sparse reward without modification.
- **Task 4 — PPO on MountainCar Continuous:** Apply Stable Baselines 3 PPO (`MlpPolicy`, `n_steps=2048`, `ent_coef=0.01`, 200,000 timesteps) to solve `MountainCarContinuous-v0`, justifying the algorithm choice against TD3, SAC, and TRPO.

#### 🔑 Key Concepts

| Concept | Description |
|---------|-------------|
| ❄️ Cold-Start Failure | Vanilla DQN never reaches the goal via random exploration — 0% success baseline |
| ⚡ Potential-Based Shaping | `F(s,s') = k_energy · (E(s') − E(s))` — preserves the true optimal policy (Ng et al., 1999) |
| 🪜 Reverse Curriculum | Initial state distribution progressively widened from near-goal to full range |
| 📈 Rolling Success Rate | 100-episode rolling window tracks curriculum convergence |
| 🤖 SB3 PPO | `MlpPolicy`, `n_steps=2048`, `ent_coef=0.01` — solves the continuous variant near-optimally |

#### 📤 Key Numerical Results

```
Env: MountainCar-v0 (DQN/shaping/curriculum) / MountainCarContinuous-v0 (PPO)

┌─────────────────────────────┬──────────────────────────────────────────────┐
│ Method                      │ Outcome                                      │
├─────────────────────────────┼──────────────────────────────────────────────┤
│ Vanilla DQN (baseline)      │ 0% success across 150 episodes (reward −200) │
│ Energy-based reward shaping │ Reaches flag within ≈150–200 episodes        │
│ Reverse curriculum learning │ ≥80–90% success by 1000 episodes, lower      │
│                             │ variance than shaping                        │
│ SB3 PPO (continuous)        │ Test reward ≈ −0.00017 (effectively optimal) │
└─────────────────────────────┴──────────────────────────────────────────────┘
```

**Key result:** Reward shaping offers faster early convergence; curriculum learning offers higher final stability with no reward corruption. PPO trivially solves the continuous variant. The lab demonstrates that the right exploration or curriculum strategy can substitute for manual reward engineering.

---

## 🔗 Skill Progression Across Labs

```
╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
║                     ROBOTICS AND REINFORCEMENT LEARNING — PROGRESSION                            ║
╠══════════════════╦═══════════════════╦═══════════════════╦═══════════════════╦═══════════════════╣
║  🔵 LAB 1 🔵    ║  🟠 LAB 2 🟠     ║  🟣 LAB 3 🟣     ║  🟢 LAB 4 🟢     ║  🔴 LAB 5 🔴     ║
║  SVD &           ║  Physics-         ║  Value- &         ║  Advanced         ║  Sparse           ║
║  Matrix Decomp.  ║  Informed NNs &   ║  Policy-Based RL  ║  Policy           ║  Reward           ║
║                  ║  ODE Discovery    ║  (DQN + PG)       ║  Optimisation     ║  Strategies       ║
║                  ║                   ║                   ║  (PPO + GAE + A-C)║  (Shaping,        ║
║                  ║                   ║                   ║                   ║  Curriculum)      ║
╠══════════════════╩═══════════════════╩═══════════════════╩═══════════════════╩═══════════════════╣
║                  Increasing complexity · Closer to real robotics deployment                      ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════╝
```

---

## ⚙️ Technical Specifications

### 🌍 Environments and Datasets

| Lab | Primary Asset | Source |
|-----|--------------|--------|
| 1 | `Safer-City-Driving.jpg` (road image) | Provided |
| 1 | `housing.data` (Boston Housing, 506 × 13) | UCI / Scikit-learn |
| 1 | `allFaces(1).mat` (Yale Face Database B, 192×168 px) | Yale Vision Lab |
| 2 | Synthetic ODE data (`y = A cos(ωx + φ)`) | Generated in-notebook |
| 2 | `train.csv` / `test.csv` (planar quadrotor trajectories) | Google Drive (gdown) |
| 3 | `CartPole-v0` / `CartPole-v1` | Gymnasium classic control |
| 4 | `Acrobot-v1` | Gymnasium classic control |
| 5 | `MountainCar-v0` / `MountainCarContinuous-v0` | Gymnasium classic control |

### 🧰 Frameworks

| Purpose | Package |
|---------|---------|
| Array math and SVD | NumPy, SciPy |
| Neural networks and autodiff | PyTorch ≥ 2.0 (`torch.func` for PINNs) |
| RL environments | Gymnasium ≥ 0.26 |
| Baseline RL algorithms | Stable Baselines 3 |
| Classical ML utilities | Scikit-learn |
| Plotting | Matplotlib |

### 💻 Hardware Requirements

All five labs are designed to run on **CPU only** in Google Colab. No GPU is required. Labs 3–5 auto-detect CUDA and use it if available, but converge fully on CPU within the estimated time budgets.

### ⏱️ Estimated Total Runtime (CPU)

| Lab | Estimated Time |
|-----|---------------|
| Lab 1 | 5–10 min |
| Lab 2 | 25–45 min |
| Lab 3 | 15–35 min |
| Lab 4 | 6–13 min |
| Lab 5 | 10–25 min |
| **Total** | **~60–130 min** |

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/umerahmedbaig7/Robotics-and-Reinforcement-Learning.git
cd Robotics-and-Reinforcement-Learning
```

### 2️⃣ Install All Dependencies Locally

```bash
pip install numpy scipy matplotlib scikit-learn scikit-image \
            torch gymnasium[classic-control] stable-baselines3 \
            pandas gdown pyvirtualdisplay
```

> 📌 All packages are pre-installed in Google Colab. Labs are designed to run without any manual installation in that environment.

### 3️⃣ Run in Google Colab

1. Open any lab notebook via Google Colab (File → Open notebook → GitHub).
2. **For Lab 1:** upload `Safer-City-Driving.jpg`, `housing.data`, and `allFaces(1).mat` to your Google Drive under a folder (e.g. `MLR_DATA`) and set `DATA_DIR` accordingly.
3. **For Lab 2:** the quadrotor CSV datasets are downloaded automatically via `gdown`.
4. **For Labs 3–5:** all data is generated or downloaded at runtime — no manual setup needed.
5. Run all cells sequentially (**Runtime → Run all**).

---

## 📊 Key Results Summary

| Lab | Method | Result |
|-----|--------|--------|
| 1 — SVD | Truncated SVD (99% energy) | Near-lossless image reconstruction at fraction of storage |
| 1 — SVD | Gavish–Donoho threshold | Correct rank-2 recovery from noisy data (vs. overestimation by ad-hoc threshold) |
| 1 — Regression | SVD pseudo-inverse on Boston Housing | Positive R² on held-out 20% test split |
| 2 — PINNs | PINN + dummy collocation | Accurate ODE extrapolation to `[-10, 10]` from 50 noisy points in `[-5, 5]` |
| 2 — Discovery | Joint optimisation with `omega_t` | True ω = 1.5 recovered from data alone |
| 2 — Quadrotor | Physics-informed model | Matches or outperforms pure data-driven baseline on all 6 state components |
| 3 — DQN | Target Network | Smoother convergence than vanilla DQN; solves CartPole-v1 (500-step limit) |
| 3 — PG | Reward-To-Go | More stable than plain REINFORCE; achieves CartPole-v1 threshold |
| 4 — PPO | GAE + clipped surrogate | Stable Acrobot swingup within 170 episodes; generalises to unseen seeds |
| 5 — Shaping | Energy-based potential | Solves MountainCar-v0 within 400 episodes from 0% baseline |
| 5 — Curriculum | Reverse curriculum | ≥ 80–90% success rate with original sparse reward; lower variance than shaping |
| 5 — PPO | Stable Baselines 3 | MountainCarContinuous-v0 total reward ≈ −0.00017 (effectively optimal) |

---

## 🧠 Analysis and Discussion

### 🔵 SVD as a Universal Tool (Lab 1)

The first lab establishes the SVD as the common language underlying regression, compression, face recognition, robust decomposition, and optimal denoising. This perspective — seeing diverse methods as instances of one algebraic structure — carries forward into all subsequent labs, where the "structured prior" changes form but the principle remains the same.

### 🟠 Why Physics Constraints Beat Black-Box Models for Scarce Data (Lab 2)

With only 50 training samples, a vanilla MLP cannot identify the correct function from the infinite hypothesis space of neural networks. The physics residual loss `𝒩[f, x] = 0` dramatically reduces this space to functions satisfying the governing equation — equivalent to injecting infinite structural prior knowledge. The dummy collocation points then extend this constraint over the full test domain at negligible cost, enabling reliable out-of-distribution generalisation unavailable to any purely data-driven approach.

### 🟣 Value-Based vs. Policy-Based RL (Lab 3)

DQN and REINFORCE represent two fundamentally different approaches to the RL objective. DQN is more sample-efficient (replay buffer reuses transitions) but requires a Q-function approximation. REINFORCE is unbiased but high-variance; RTG reduces variance by attributing each action only to its causal future. On CartPole, both approaches converge, but the architectural choices (target network for DQN, RTG + entropy for PG) are critical for stability.

### 🟢 GAE and the PPO Trust Region (Lab 4)

GAE with λ = 0.95 captures roughly 20 steps of future credit assignment — well-matched to Acrobot's preparatory swing structure. The PPO clip at ε = 0.3 allows larger per-step updates than the canonical 0.2, trading some stability for faster convergence on the 170-episode budget. The separate policy and value optimisers prevent value regression gradients from contaminating policy improvement.

### 🔴 Reward Engineering vs. Curriculum Learning (Lab 5)

Both reward shaping and curriculum learning address the identical root cause — absence of a gradient signal toward the goal — but through different mechanisms. Shaping injects goal information into the reward function (faster early convergence, higher variance, risk of reward corruption). Curriculum restructures the experience distribution (slower start, lower variance, preserves task integrity). For deployment-critical systems where the reward function must faithfully represent the true objective, curriculum learning is the more principled choice.

---

## ⚠️ Notes and Limitations

- **Stochastic results:** Labs 1 (LASSO), 3, 4, and 5 do not fix all random seeds. Qualitative conclusions are consistent across runs; specific numerical results will vary.
- **`allFaces(1)` filename:** The parentheses in the `.mat` filename may cause path issues on Windows. Rename to `allFaces1.mat` and update `loadmat` accordingly.
- **`torch.func` API:** Lab 2 requires PyTorch ≥ 2.0 for `jacrev`, `hessian`, and `vmap`. On older installs, the legacy `functorch` package provides equivalent functionality.
- **Lab 2, Task 6 cell ordering:** Cell 22 references `cutoff` from Cell 20; these two cells must be run together, or `cutoff` must be defined manually before running Cell 22.
- **Lab 4 `clip_eps = 0.3`:** Slightly wider than the canonical 0.2, chosen to accelerate convergence within the 170-episode training budget.
- **Lab 5 Task 3 naming:** The `train_dqn_curriculum_free()` function implements progressively widened initialisation despite the "free" label — the name reflects the absence of an explicit curriculum scheduler class.
- **Stable Baselines 3 version:** Lab 5's `ts2xy` import path requires SB3 ≥ 1.7.

---

## 🧰 Tech Stack

<div align="center">

| 🛠️ Tool | 🔖 Version | 🎯 Role in This Course | 🧪 Used In |
|:-------:|:---------:|:---------------------:|:----------:|
| ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) | 3.8+ | Core language — all labs | All |
| ![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?logo=pytorch&logoColor=white) | ≥ 2.0 | Neural networks, autodiff, `torch.func` for PINNs | Labs 2–4 |
| ![Gymnasium](https://img.shields.io/badge/-Gymnasium-00C853?logo=openaigym&logoColor=white) | ≥ 0.26 | RL environments — classic control suite | Labs 3–5 |
| ![StableBaselines3](https://img.shields.io/badge/-Stable--Baselines3-7B2FBE) | Required | Baseline PPO implementation | Lab 5 |
| ![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?logo=scikitlearn&logoColor=white) | Required | Classical ML utilities, regression metrics | Lab 1 |
| ![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white) | Required | Array math, SVD, matrix ops | All |
| ![SciPy](https://img.shields.io/badge/-SciPy-8CAAE6?logo=scipy&logoColor=white) | Required | Linear algebra, signal processing | Lab 1 |
| ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?logo=python&logoColor=white) | Required | Plotting across all labs | All |

</div>

**No robotics toolboxes beyond Gymnasium and Stable-Baselines3.** Every core algorithm — from the SVD-based decompositions to the PPO clipped surrogate — is implemented explicitly in Python/PyTorch, ensuring depth of understanding at every level of the learning stack.

---

## 👤 Author

<div align="center">

### Umer Ahmed Baig Mughal

🎓 **MSc Robotics and Artificial Intelligence** — ITMO University <br>
🏛️ *Faculty of Control Systems and Robotics* <br>
🔬 *Specialization: Machine Learning · Computer Vision · Human-Robot Interaction · Autonomous Systems · Robotic Motion Control*

[![GitHub](https://img.shields.io/badge/GitHub-umerahmedbaig7-181717?style=for-the-badge&logo=github)](https://github.com/umerahmedbaig7)

</div>

---

## 📄 License

All lab notebooks and associated code in this repository are intended for **academic and research use**. They were developed as part of the *Robotics and Reinforcement Learning* and *Machine Learning for Robotics* courses within the MSc Robotics and Artificial Intelligence programme at ITMO University. Redistribution, modification, and use in derivative academic work are permitted with appropriate attribution to the original author.

---

<div align="center">

*Robotics and Reinforcement Learning | MSc Robotics and Artificial Intelligence | ITMO University*

⭐ *If this repository helped you understand robotics and reinforcement learning, consider giving it a star!* ⭐

</div>
