# Lab 2 — Physically Consistent Deep Learning: PINNs, ODE Discovery & Quadrotor Dynamics

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![torch.func](https://img.shields.io/badge/torch.func-JAX--style-lightgrey)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4%2B-green)
![Field](https://img.shields.io/badge/Field-Physics--Informed%20ML-purple)
![License](https://img.shields.io/badge/License-Academic-yellow)

> **Course:** Machine Learning for Robotics — Faculty of Control Systems and Robotics, ITMO University
> **Author:** Umer Ahmed Baig Mughal — MSc Robotics and Artificial Intelligence
> **Topic:** PINNs · ODE Solving · ODE Discovery · Physics Constraints · Quadrotor Dynamics · Dummy Data Regularisation · Learnable Physical Parameters

---

## Table of Contents

1. [Objective](#objective)
2. [Theoretical Background](#theoretical-background)
   - [Physics-Informed Neural Networks (PINNs)](#physics-informed-neural-networks-pinns)
   - [Black-Box Deep Learning vs. PINNs](#black-box-deep-learning-vs-pinns)
   - [Automatic Differentiation via torch.func](#automatic-differentiation-via-torchfunc)
   - [Dummy Data Physics Regularisation](#dummy-data-physics-regularisation)
   - [ODE Discovery with Learnable Parameters](#ode-discovery-with-learnable-parameters)
   - [Periodic Boundary Constraint](#periodic-boundary-constraint)
   - [Quadrotor Planar Dynamics Model](#quadrotor-planar-dynamics-model)
   - [System Properties](#system-properties)
3. [Tasks and Implementation](#tasks-and-implementation)
   - [Task 1 — ODE Solution: Known Physics (Black-Box vs. PINN)](#task-1--ode-solution-known-physics-black-box-vs-pinn)
   - [HW1 — Periodic Physical Constraint](#hw1--periodic-physical-constraint)
   - [Task 2 — ODE Discovery: Unknown Frequency ω](#task-2--ode-discovery-unknown-frequency-ω)
   - [HW2 — ODE Discovery with Dummy Data](#hw2--ode-discovery-with-dummy-data)
   - [HW3 — ODE Discovery: Unknown Frequency + Unknown Bias](#hw3--ode-discovery-unknown-frequency--unknown-bias)
   - [Task 3 — Quadrotor Planar Dynamics: Data-Driven Model](#task-3--quadrotor-planar-dynamics-data-driven-model)
   - [Task 4 — Quadrotor Planar Dynamics: Physics-Informed Model](#task-4--quadrotor-planar-dynamics-physics-informed-model)
4. [Validation and Metrics](#validation-and-metrics)
5. [System Parameters](#system-parameters)
6. [Implementation](#implementation)
   - [File Structure](#file-structure)
   - [Function Reference](#function-reference)
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

This lab implements and evaluates Physics-Informed Neural Networks (PINNs) as a framework for incorporating physical constraints into deep learning models. The lab progresses from vanilla black-box deep learning through physics-constrained models, ODE discovery with learnable parameters, and ultimately to a physically consistent model of a planar quadrotor system.

The key learning outcomes are:

- Training a standard MLP on sparse, noisy data for the ODE `d²y/dx² + y = 0` and observing overfitting and poor generalisation to out-of-distribution test intervals.
- Constructing a PINN by augmenting the MSE data loss with a physics residual loss that penalises violations of the governing differential equation, and observing its regularisation effect even on noisy 50-point training sets.
- Leveraging dummy collocation points — unlabelled x-values at which the physics loss is enforced — to dramatically extend the spatial domain over which the ODE constraint is active, enabling reliable extrapolation to the wider `[-10, 10]` test interval from a `[-5, 5]` training interval.
- Implementing a periodic boundary condition as an additional physics-based loss term by enforcing `f(-π) = f(π)` and `f'(-π) = f'(π)`, matching the known 2π-periodicity of the cosine solution.
- Performing ODE discovery by introducing `ω` as a jointly optimised learnable scalar parameter alongside the network weights, allowing the PINN to simultaneously solve and identify the governing equation `d²y/dx² + ω²y = 0` from data alone.
- Extending ODE discovery to the case of an unknown bias term (`y(x) = A cos(ωx + φ) + B`), requiring derivation of the third-order DE operator `N` and implementation of a multi-parameter discovery PINN.
- Implementing a data-driven neural network for 6-DOF planar quadrotor state derivative prediction (`ẏ, ż, φ̇, v̇y, v̇z, ω̇`) from state-control input pairs `(x, u)` using real CSV trajectory data.
- Building a physics-informed quadrotor model by incorporating the known analytical equations of motion as an additional loss term and benchmarking against both the pure data-driven model and the analytical model.

The lab is implemented as a single Google Colab Jupyter notebook (`PINNs_ODE_Discovery_Quadrotor_Dynamics.ipynb`) executed on CPU, producing loss convergence curves, train/test prediction plots, ODE discovery parameter traces, and state-wise error comparisons for the quadrotor system.

---

## Theoretical Background

### Physics-Informed Neural Networks (PINNs)

A Physics-Informed Neural Network augments the standard supervised loss with a physics residual term that penalises violations of the governing physical law. For an ODE of the form `𝒩[y, x] = 0`, the total loss is:

```
L_total = L_data + λ · L_physics

where:
    L_data   = MSE(f_NN(X_train), Y_train)          supervised data loss
    L_physics = MSE(𝒩[f_NN, X_colloc], 0)            physics residual on collocation points
    λ        = coefficient controlling physics weight (coeff_phys)
```

For the harmonic oscillator ODE `d²y/dx² + y = 0`, the physics loss becomes:

```
L_physics = MSE(d²f_NN/dx²(X), -f_NN(X))
```

This formulation does not require labels at collocation points — only the x-coordinates are needed, making the physics constraint label-free and applicable to arbitrary domains.

### Black-Box Deep Learning vs. PINNs

Standard MLPs trained on sparse, noisy data exhibit two characteristic failure modes that PINNs directly address:

| Failure Mode | Black-Box MLP | PINN |
|---|---|---|
| **Overfitting** | Memorises noise in training data; training loss ≪ test loss | Physics constraint acts as a structured regulariser, preventing arbitrary function fitting |
| **Out-of-distribution generalisation** | Performance degrades sharply outside the training interval | Dummy-data physics loss enforces the correct functional form globally, enabling reliable extrapolation |

The physics constraint is not merely a regulariser — it encodes prior knowledge about the solution manifold, reducing the effective hypothesis space to functions that satisfy the governing equation.

### Automatic Differentiation via torch.func

PINNs require computing derivatives of the neural network output with respect to its input. This lab uses PyTorch's functional API, which mirrors the JAX functional programming style:

```python
from torch.func import jacrev, hessian, vmap

f       = lambda x: model(x)           # network as a pure function
df_dx   = vmap(jacrev(f))              # first derivative, batched
d2f_dx2 = vmap(hessian(f))             # second derivative, batched
```

- **`jacrev`** computes the Jacobian using reverse-mode autodiff — equivalent to a single `backward()` call.
- **`hessian`** computes the Hessian as the Jacobian of the Jacobian — two nested autodiff passes.
- **`vmap`** vectorises a scalar-input function over a batch of inputs, replacing an explicit `for` loop with a single parallelised operation.

This functional approach is necessary because `jacrev`, `hessian`, and `vmap` accept pure Python functions, not `nn.Module` objects. The `lambda x: model(x)` wrapper provides the required callable interface.

### Dummy Data Physics Regularisation

Standard PINNs apply the physics loss only at the training data locations `X_train`. This limits the spatial extent of the physics constraint to the training interval `[-5, 5]`. Dummy data extends this:

```
X_dummy = arange(-10, 10, step)        # dense collocation points, no Y required

L_phys2 = MSE(d²f_NN/dx²(X_dummy), -f_NN(X_dummy))

L_total = L_data + λ₁·L_phys1(X_train) + λ₂·L_phys2(X_dummy)
```

The dummy points do not require ground-truth labels — the physics residual is computed from the network's own output and its second derivative. By enforcing `𝒩[f_NN, x] = 0` over a dense grid covering the full test interval, the network is constrained to satisfy the ODE everywhere, enabling accurate extrapolation beyond the training range.

### ODE Discovery with Learnable Parameters

When the governing equation contains unknown parameters, those parameters can be jointly optimised alongside the network weights. For the generalised harmonic oscillator `d²y/dx² + ω²y = 0` with unknown `ω`:

```python
omega_t = torch.tensor(0.75, requires_grad=True)    # learnable scalar

params = [*model.parameters(), omega_t]             # joint parameter set

loss_phys = MSE(d2f_dx2(X_train), -omega_t**2 * f(X_train))
```

Gradients flow through `omega_t` during `loss.backward()`, and the Adam optimiser updates it alongside the network weights. The true value `ω = 1.5` is recovered over training — this is a form of system identification embedded within the PINN training loop.

### Periodic Boundary Constraint

The cosine solution `y(x) = A cos(x + φ)` is 2π-periodic. This can be explicitly enforced as a boundary condition loss:

```
L_periodic = MSE(f(-π), f(π)) + MSE(f'(-π), f'(π))

L_total = L_data + λ₁·L_phys1 + λ₂·L_phys2 + λ_periodic·L_periodic
```

The first term matches function values at the period boundaries; the second matches first derivatives (ensuring smooth periodic continuation). This enforces a global structural property of the solution that the local ODE constraint alone cannot guarantee.

### Quadrotor Planar Dynamics Model

The planar (2D) quadrotor is modelled as a rigid body with 6 state variables and 2 control inputs. The known analytical equations of motion are:

```
State:   x = [y, z, φ, vy, vz, ω]ᵀ       (position, attitude, linear and angular velocities)
Control: u = [fL, fR]ᵀ                    (left and right rotor forces)

ẏ     = vy
ż     = vz
φ̇     = ω
v̇y    = -(fL + fR) / m · sin(φ)
v̇z    = (fL + fR) / m · cos(φ) − g
ω̇     = L · (fR − fL) / I_xx

Physical constants:
    m     = 0.35 kg        (vehicle mass)
    I_xx  = 0.005 kg·m²    (moment of inertia)
    L     = 0.1 m          (arm length)
    g     = 9.81 m/s²      (gravitational acceleration)
```

This analytical model `f_analyt(x, u)` serves as the physics teacher for the PINN: the physics loss penalises deviation of the neural network's predictions from what the analytical equations dictate.

### System Properties

| Property | Value | Notes |
|---|---|---|
| ODE (Task 1 / HW) | `d²y/dx² + y = 0` | Solution: `y = A cos(x + φ)` |
| ODE (Task 2 / HW2–3) | `d²y/dx² + ω²y = 0` | Solution: `y = A cos(ωx + φ)`, ω unknown |
| Training interval | `[-5, 5]` | 50 noisy samples |
| Test interval | `[-10, 10]` | 1000 samples, wider than training |
| Noise level | σ = 0.1 | Added to both X and Y |
| MLP architecture | `[1 → 64 → 64 → 1]` | GELU activations throughout |
| Optimiser | Adam, lr = 0.01 | CosineAnnealingLR scheduler |
| LR scheduler period | `num_epochs / 6` | Helps escape gradient local minima |
| Quadrotor state dim | 6 | y, z, φ, vy, vz, ω |
| Quadrotor control dim | 2 | fL, fR (rotor forces) |
| Quadrotor MLP | `[8 → 64 → 64 → 6]` | Input: state + control concatenated |
| CSV time step | dt = 0.01 s | Derivatives approximated as Δx / Δt |
| Training data subsampling | 1 in 10 rows | `frac_samples = 10` |

---

## Tasks and Implementation

### Task 1 — ODE Solution: Known Physics (Black-Box vs. PINN)

**Goal:** Train a black-box MLP and a PINN on sparse noisy data for `d²y/dx² + y = 0`, and compare generalisation.

**Black-Box baseline (5000 epochs, data loss only):**

```python
Y_pred   = f(X_train)
loss_mse = loss_fun(Y_train, Y_pred)
loss     = loss_mse
```

Exhibits overfitting on training data and poor test performance outside `[-5, 5]`.

**PINN 1st try (1000 epochs, physics on training points):**

```python
Y_pred        = f(X_train)
d2Y_dX2_pred  = d2f_dx2(X_train).reshape(-1, 1)
loss_mse      = loss_fun(Y_train, Y_pred)
loss_phys     = loss_fun(d2Y_dX2_pred, -Y_pred)          # 𝒩[f,x] = d²f/dx² + f = 0
loss          = loss_mse + coeff_phys * loss_phys
```

Reduces overfitting. Physics constraint regularises the solution.

**PINN 2nd try (5000 epochs, physics on training + dummy points):**

```python
X_dummy           = torch.arange(-10, 10, 0.01).reshape(-1, 1)
Y_pred_dummy      = f(X_dummy)
d2Y_dX2_dummy     = d2f_dx2(X_dummy).reshape(-1, 1)
loss_phys2        = loss_fun(d2Y_dX2_dummy, -Y_pred_dummy)
loss              = loss_mse + coeff_phys1 * loss_phys1 + coeff_phys2 * loss_phys2
```

Accurate extrapolation to `[-10, 10]` from 50 noisy points in `[-5, 5]`.

---

### HW1 — Periodic Physical Constraint

**Goal:** Propose and implement an additional physics-based constraint exploiting the 2π-periodicity of the cosine solution.

```python
x_left  = torch.tensor([[-torch.pi]])
x_right = torch.tensor([[torch.pi]])

y_left,  y_right  = f(x_left),  f(x_right)
dy_left, dy_right = dy_dx(x_left), dy_dx(x_right)

loss_periodic = loss_fun(y_left, y_right) + loss_fun(dy_left, dy_right)

loss = (loss_mse
      + coeff_phys1 * loss_phys1
      + coeff_phys2 * loss_phys2
      + coeff_periodic * loss_periodic)
```

The periodic constraint anchors the model to the correct global structure beyond what the local ODE constraint provides.

---

### Task 2 — ODE Discovery: Unknown Frequency ω

**Goal:** Simultaneously solve the ODE and discover the unknown parameter `ω = 1.5` from data.

```python
omega_t = torch.tensor(0.75, requires_grad=True)           # initial guess ≠ true value

params  = [*model.parameters(), omega_t]                   # joint optimisation

loss_phys = loss_fun(d2f_dx2(X_train).reshape(-1,1),
                     -omega_t * omega_t * Y_pred)           # 𝒩[f,x;ω] = d²f/dx² + ω²f = 0

loss = loss_mse + coeff_phys * loss_phys
```

`omega_t` is updated by Adam at each step via autograd through the physics loss. The learnt `ω` converges toward the true value of 1.5 during training.

---

### HW2 — ODE Discovery with Dummy Data

**Goal:** Extend Task 2 with dummy collocation points to improve generalisation during parameter discovery.

```python
X_dummy = torch.arange(-10, 10, 0.01).reshape(-1, 1)

Y_dummy_pred   = f(X_dummy)
d2Y_dummy_pred = d2f_dx2(X_dummy).reshape(-1, 1)
loss_phys      = loss_fun(d2Y_dummy_pred, -omega_t * omega_t * Y_dummy_pred)

loss = loss_mse + coeff_phys * loss_phys
```

Enforcing the ODE with the learnable `ω` over a dense dummy grid simultaneously improves both parameter discovery accuracy and out-of-distribution generalisation.

---

### HW3 — ODE Discovery: Unknown Frequency + Unknown Bias

**Goal:** Derive the DE operator for `y(x) = A cos(ωx + φ) + B` and implement a PINN that discovers both `ω` and `B`.

**Derivation:** For `y = A cos(ωx + φ) + B`:

```
dy/dx   =  -Aω sin(ωx + φ)
d²y/dx² =  -Aω² cos(ωx + φ)  =  -ω²(y − B)

∴  d²y/dx² + ω²y − ω²B = 0        ← the governing DE operator 𝒩[y, x; ω, B]
```

Two learnable scalars `omega_t` and `B_t` are jointly optimised with the network weights using the modified physics residual.

---

### Task 3 — Quadrotor Planar Dynamics: Data-Driven Model

**Goal:** Train a standard MLP to predict state derivatives `ẋ = [ẏ, ż, φ̇, v̇y, v̇z, ω̇]` from `(x, u)` pairs using real CSV trajectory data.

```python
class Model(nn.Module):
    def forward(self, x, u):
        inp = torch.cat([x, u], dim=1)    # (B, 8): 6 states + 2 controls
        return self.net(inp)               # (B, 6): 6 state derivatives

# Derivative approximation from discrete data
dx = (x_next - x) / 0.01    # Δx / Δt

# Data loss only
x_dot_pred = model(x_train, u_train)
loss = loss_fun(x_dot_pred, dX_train)
```

State-wise MSE is reported for each of the 6 state components.

---

### Task 4 — Quadrotor Planar Dynamics: Physics-Informed Model

**Goal:** Incorporate the known analytical model as a physics teacher loss and compare against the data-driven baseline and the analytical model.

```python
def f_anlyt(x, u):
    y, z, phi, vy, vz, omega = x[:,0], ..., x[:,5]
    fL, fR = u[:,0], u[:,1]
    F_total = fL + fR
    # ...analytical equations of motion...
    return torch.stack([ydot, zdot, phidot, vydot, vzdot, omegadot], dim=1)

# Physics loss: NN predictions should match the analytical model
x_dot_nn    = model(x_train, u_train)
x_dot_anlyt = f_anlyt(x_train, u_train)
loss_phys   = loss_fun(x_dot_nn, x_dot_anlyt)

loss = loss_data + coeff_phys * loss_phys
```

The analytical model acts as a soft constraint rather than a hard constraint — the network can learn corrections to the analytical model while still being guided by it.

---

## Validation and Metrics

### ODE Tasks — Generalisation Quality

Reconstruction quality is assessed by the MSE between the predicted function `f_NN(X_test)` and the ground-truth `Y_test` over the wider `[-10, 10]` interval. Visual assessment of prediction vs. actual plots confirms whether the model has generalised beyond the training range.

### ODE Discovery — Parameter Recovery

The learnt `omega_t` value at the end of training is compared against the true `ω = 1.5`. Convergence is tracked by printing `omega_t.item()` every 100 epochs during the training loop.

### Quadrotor — State-wise MSE

| Metric | Formula | Interpretation |
|---|---|---|
| MSE (PINN) | `mean((ẋ_pred − ẋ_true)²)` | Overall derivative prediction error |
| MSE (Analytical) | `mean((f_anlyt − ẋ_true)²)` | Baseline: pure physics model error |
| State MAE | `mean(|ẋ_pred − ẋ_true|, dim=0)` | Per-component: ẏ, ż, φ̇, v̇y, v̇z, ω̇ |

State-wise MAE isolates which components the model captures well (kinematic states: ẏ, ż, φ̇) versus those requiring accurate force knowledge (dynamic states: v̇y, v̇z, ω̇).

---

## System Parameters

### ODE Experiment Parameters

| Parameter | Task 1 (BB) | Task 1 (PINN) | Task 1 (PINN + Dummy) | Task 2 / HW2 |
|---|---|---|---|---|
| `num_epochs` | 5000 | 1000 | 5000 | 3000 |
| `lr` | 0.01 | 0.01 | 0.01 | 0.01 |
| `coeff_phys` | — | 1.0 | 1.0 / 1.0 | 2.5 |
| `coeff_periodic` | — | — | — | — |
| Dummy range | — | — | `[-10, 10]` step 0.01 | `[-10, 10]` step 0.01 |
| `omega_t` init | — | — | — | 0.01 |
| True ω | — | — | — | 1.5 |

### Quadrotor Physical Constants

| Parameter | Value | Description |
|---|---|---|
| `m` | 0.35 kg | Vehicle mass |
| `I_xx` | 0.005 kg·m² | Moment of inertia about roll axis |
| `L` | 0.1 m | Rotor arm length |
| `g` | 9.81 m/s² | Gravitational acceleration |
| `dt` | 0.01 s | Simulation time step for derivative approximation |

### Quadrotor Training Parameters

| Parameter | Value | Description |
|---|---|---|
| MLP input dim | 8 | 6 state + 2 control |
| MLP output dim | 6 | 6 state derivatives |
| Hidden layers | `[64, 64]` | GELU activations |
| `num_epochs` | 5000 | Training iterations |
| `lr` | 0.001 | Adam learning rate |
| `frac_samples` | 10 | 1-in-10 subsampling of training CSV |
| Noise level | σ = 0.1 | Added to X_train and Y_train |
| `coeff_phys` | 1.0 | Physics loss weight |

---

## Implementation

### File Structure

```
Lab2/
├── Readme.md
└── PINNs_ODE_Discovery_Quadrotor_Dynamics.ipynb    # Complete lab — all tasks and HW
```

Data files (downloaded at runtime via `gdown`):

| File | Type | Used In |
|---|---|---|
| `train.csv` | CSV trajectory data | Tasks 3, 4 |
| `test.csv` | CSV trajectory data | Tasks 3, 4 |

---

### Function Reference

#### `generate_data(interval, num_samples, A, phi, omega)` — synthetic ODE dataset

Generates labelled `(X, Y)` pairs for `y = A cos(ωx + φ)` and verifies the ODE holds on the generated data using autograd.

| Argument | Type | Default | Description |
|---|---|---|---|
| `interval` | tuple | `(-π, π)` | Sampling interval for x |
| `num_samples` | int | 1000 | Number of samples |
| `A` | float | 1.0 | Amplitude |
| `phi` | float | 0.0 | Phase offset |
| `omega` | float | 1.5 | Angular frequency (Task 2 only) |

Returns: `(X, Y)` — tensors with `requires_grad=True` on X.

---

#### `get_model_and_derivatives(hidden_channels, activation, device)` — MLP factory

Constructs a fully-connected MLP and returns it alongside functional derivative operators.

```python
(f, df_dx, d2f_dx2), model.parameters() = get_model_and_derivatives()
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `hidden_channels` | tuple | `(64, 64)` | Hidden layer widths |
| `activation` | nn.Module | `nn.GELU` | Activation class |
| `device` | str | `"cpu"` | Compute device |

Returns: `((f, df_dx, d2f_dx2), params)` — functional API operators and trainable parameters.

---

#### `test_model(f, X, Y)` — evaluation and visualisation

Computes MSE and plots predicted vs. actual curves sorted by x.

```python
def test_model(f, X, Y):
    print(f"MSE: {F.mse_loss(f(X), Y)}")
    # sorts by X for clean curve display
    plt.plot(X_sorted, Y_sorted, label='Act')
    plt.plot(X_sorted, f(X_sorted), label='Pred')
```

---

#### `f_anlyt(x, u)` — quadrotor analytical model

Implements the planar quadrotor equations of motion as a differentiable PyTorch function.

```python
def f_anlyt(x, u):
    m, I_xx, L, g = 0.35, 0.005, 0.1, 9.81
    y, z, phi, vy, vz, omega = x[:,0], ..., x[:,5]
    fL, fR = u[:,0], u[:,1]
    F_total = fL + fR
    # ...
    return torch.stack([ydot, zdot, phidot, vydot, vzdot, omegadot], dim=1)
```

| Argument | Type | Description |
|---|---|---|
| `x` | Tensor (B, 6) | State batch: `[y, z, φ, vy, vz, ω]` |
| `u` | Tensor (B, 2) | Control batch: `[fL, fR]` |

Returns: `x_dot` — Tensor (B, 6) of state derivatives.

---

### Algorithm Walkthrough

Complete pipeline (`PINNs_ODE_Discovery_Quadrotor_Dynamics.ipynb`):

```
1. ENVIRONMENT SETUP:
       import torch, torch.func (jacrev, hessian, vmap)
       import matplotlib, numpy

2. ODE SOLUTION — Task 1:
       a. generate_train_test([-5,5], 50, [-10,10], 1000) → X_train, Y_train, X_test, Y_test
       b. Add noise: X += 0.1·randn, Y += 0.1·randn
       c. Black-box MLP (5000 epochs, MSE only) → test_model on train and test
       d. PINN 1st try (1000 epochs, physics on X_train) → test_model
       e. PINN 2nd try (5000 epochs, physics on X_train + X_dummy[-10,10]) → test_model

3. PERIODIC CONSTRAINT — HW1:
       a. Extend PINN 2nd try with L_periodic = MSE(f(-π), f(π)) + MSE(f'(-π), f'(π))
       b. coeff_periodic = 0.5
       c. test_model on train and test

4. ODE DISCOVERY — Task 2:
       a. generate_train_test with omega=1.5
       b. Initialise omega_t = 0.75 (requires_grad=True)
       c. Joint optimisation: params = [*model.params, omega_t]
       d. Physics loss: MSE(d²f/dx², -omega_t² · f)
       e. Log omega_t every 100 epochs → observe convergence to 1.5

5. ODE DISCOVERY + DUMMY DATA — HW2:
       a. omega_t = 0.01 (harder initialisation)
       b. X_dummy = arange(-10, 10, 0.01)
       c. Physics loss on dummy: MSE(d²f_dummy/dx², -omega_t² · f_dummy)
       d. 3000 epochs, coeff_phys=2.5 → test_model

6. ODE DISCOVERY + UNKNOWN BIAS — HW3:
       a. Derive 𝒩[y,x;ω,B]: d²y/dx² + ω²y - ω²B = 0
       b. Initialise omega_t, B_t with requires_grad=True
       c. Physics loss: MSE(d²f/dx² + omega_t²·f, omega_t²·B_t)
       d. Joint optimisation of network + omega_t + B_t

7. QUADROTOR DATA-DRIVEN — Task 3:
       a. gdown → train.csv, test.csv
       b. Compute dx = (x_next - x) / 0.01
       c. Subsample 1:10, add noise to X_train and Y_train
       d. Model: cat([x,u]) → [64,64] GELU MLP → 6 outputs
       e. 5000 epochs, MSE on state derivatives
       f. State-wise MAE evaluation and 2×3 prediction subplot

8. QUADROTOR PINN — Task 4:
       a. Physics loss: MSE(model(x,u), f_anlyt(x,u))
       b. Total loss = L_data + coeff_phys · L_phys
       c. Compare MSE: PINN vs. data-driven vs. analytical model
       d. State-wise comparison table
```

---

## How to Run

### Open in Google Colab

The notebook is designed for Google Colab with CPU execution. No GPU is required.

1. Upload `PINNs_ODE_Discovery_Quadrotor_Dynamics.ipynb` to Google Drive or GitHub.
2. Open with Google Colab.
3. The quadrotor CSV datasets are downloaded automatically via `gdown` — no manual upload is needed.
4. Run all cells sequentially (`Runtime → Run all`).

### Install Dependencies

All required packages are pre-installed in Google Colab. For a local environment:

```bash
pip install torch numpy matplotlib pandas gdown
```

### Dataset Setup — Quadrotor CSV Data

The training and test datasets are downloaded at runtime:

```python
import gdown
gdown.download('https://drive.google.com/file/d/1mdCwjWupljOQH4ySLUxuTytD1Hsrzfho/view?usp=sharing', fuzzy=True)
gdown.download('https://drive.google.com/file/d/1LngllwyXvPtWD7vB-cceaFibf0UDY7N4/view?usp=sharing', fuzzy=True)
```

If the Google Drive links are unavailable, place `train.csv` and `test.csv` manually in `/content/` and comment out the `gdown` cells.

### Estimated Execution Time

| Section | Estimated Time |
|---|---|
| Task 1 — Black-box (5000 epochs) | 2–4 min |
| Task 1 — PINN 1st try (1000 epochs) | < 1 min |
| Task 1 — PINN 2nd try (5000 epochs + dummy) | 5–8 min |
| HW1 — Periodic PINN (5000 epochs) | 5–8 min |
| Task 2 — ODE discovery (2500 epochs) | 1–2 min |
| HW2 — ODE discovery + dummy (3000 epochs) | 3–5 min |
| HW3 — Unknown ω + B | 3–5 min |
| Task 3 — Quadrotor data-driven (5000 epochs) | 3–5 min |
| Task 4 — Quadrotor PINN (5000 epochs) | 3–5 min |
| **Total** | **~25–45 min** |

### Modifying Experiment Parameters

```python
# Task 1 — change physics loss weight
coeff_phys = 1.0          # increase for stronger physics regularisation

# Task 1 — change dummy data density
X_dummy = torch.arange(-10, 10, 0.01)   # finer grid → stronger physics enforcement

# Task 2 — change omega initialisation
omega_t = torch.tensor(0.75, requires_grad=True)    # farther from 1.5 = harder discovery

# Quadrotor — change subsampling
frac_samples = 10          # lower = more data; higher = sparser training set

# Quadrotor — change physics loss weight
coeff_phys = 1.0           # increase to bias model closer to analytical equations
```

---

## Results

### Task 1 — ODE Solution Comparison

| Model | Train MSE | Test MSE ([-10,10]) | Generalisation |
|---|---|---|---|
| Black-box MLP | Low (overfits) | High | Poor — fails outside `[-5,5]` |
| PINN (physics on train) | Low | Moderate | Improved — physics regularises fitting |
| PINN + dummy data | Low | Low | Excellent — correctly extrapolates to `[-10,10]` |

The PINN with dummy data correctly recovers the cosine solution even from 50 noisy samples limited to `[-5, 5]`, demonstrating data-efficient physically-consistent learning.

### HW1 — Periodic Constraint

Adding the periodic boundary loss further stabilises the solution by anchoring it to the correct 2π-period structure. The periodic constraint most visibly benefits cases where the dummy physics loss alone is insufficient to fully constrain the solution phase.

### Task 2 — ODE Discovery

The learnable `omega_t` converges from the initialised value of 0.75 toward the true value of 1.5 during training. Joint optimisation of physical parameters alongside network weights is a key capability of PINNs not available in standard black-box approaches.

| Method | `omega_t` (start) | `omega_t` (final) | True ω |
|---|---|---|---|
| PINN 1st try | 0.75 | ~1.5 | 1.5 |
| HW2 PINN + dummy | 0.01 | ~1.5 | 1.5 |

### Task 3 & 4 — Quadrotor Dynamics

The PINN model consistently matches or outperforms the pure data-driven model by leveraging the analytical equations of motion as a physics prior. The analytical model provides a strong baseline for the kinematic states (`ẏ, ż, φ̇`) but has fixed residual error on dynamic states (`v̇y, v̇z, ω̇`) when real-world data deviates from the idealised model.

| Component | Data-Driven MSE | PINN MSE | Analytical MSE |
|---|---|---|---|
| `dy/dt` | — | Lower | — |
| `dz/dt` | — | Lower | — |
| `dφ/dt` | — | Comparable | — |
| `dvy/dt` | — | Improved | Baseline |
| `dvz/dt` | — | Improved | Baseline |
| `dω/dt` | — | Improved | Baseline |

---

## Analysis and Discussion

### Why Physics Constraints Help with Scarce Data

With only 50 training samples, a vanilla MLP has insufficient data to identify the correct function from the infinite hypothesis space of neural networks. The physics constraint dramatically reduces this space: only functions satisfying `d²f/dx² + f = 0` are admissible. This is equivalent to supplying an infinite amount of structural prior knowledge about the solution, making PINNs exceptionally data-efficient for problems governed by known physical laws.

### The Role of Collocation Points

Standard supervised learning requires `(x, y)` pairs at every point where the model is trained. PINNs decouple supervision from physics enforcement: the data loss requires labels, but the physics loss only requires x-coordinates. This means the physics constraint can be applied at arbitrarily dense collocation grids — including regions where no data exists — at negligible cost. The dummy data results demonstrate that this directly translates to out-of-distribution generalisation, a capability fundamentally unavailable to black-box models.

### PINNs as a System Identification Framework

The ODE discovery experiments show that PINNs are not limited to solving known equations — they can identify unknown physical parameters by treating them as learnable scalars in the physics loss. This unifies system identification and state estimation in a single differentiable optimisation problem. The extension to unknown bias (HW3) illustrates how more complex unknown structures require more careful operator derivation but follow the same pattern.

### Physics Loss as Analytical Model Correction

For the quadrotor, the analytical model `f_anlyt` is known but imperfect — it omits aerodynamic drag, motor dynamics, and other real-world effects. Using it as a soft physics constraint allows the neural network to learn corrections to the analytical model from data, rather than learning the full dynamics from scratch. This hybrid approach is more data-efficient than a pure data-driven model and more accurate than the analytical model alone.

### Limitations of the Physics Residual Approach

The physics loss weight `coeff_phys` must be tuned carefully. Too small: physics constraint is ineffective. Too large: the model fits the physics at the expense of the data, effectively becoming the analytical model. The optimal balance depends on data quality, noise level, and the accuracy of the physical model. This sensitivity is a practical limitation of PINNs compared to hard-constraint methods.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| Python | ≥ 3.8 | Runtime environment |
| torch | ≥ 2.0 | Neural network training, `torch.func` API for autodiff |
| torch.func | Included with PyTorch 2.0+ | `jacrev`, `hessian`, `vmap` functional operators |
| numpy | ≥ 1.21 | Array utilities |
| matplotlib | ≥ 3.4 | Loss curves, prediction vs. actual plots |
| pandas | ≥ 1.3 | Loading quadrotor CSV datasets |
| gdown | ≥ 4.0 | Downloading datasets from Google Drive |

Install all dependencies locally:

```bash
pip install torch numpy matplotlib pandas gdown
```

All packages are pre-installed in Google Colab. The notebook is designed to run without any manual installation in the Colab environment.

---

## Notes and Limitations

- **`torch.func` vs `functorch`:** Some cells use `from functorch import vmap, jacrev` (legacy) while others use `from torch.func import jacrev, hessian, vmap` (modern). Both work in PyTorch ≥ 2.0; `functorch` is the older separate package. For consistency in a fresh environment, prefer `from torch.func import ...`.
- **`retain_graph=True` in `loss.backward()`:** Required because the physics loss reuses the same computational graph as the data loss — the second-derivative operators traverse the graph twice. Without it, PyTorch frees the graph after the first backward pass and the second fails. This increases memory usage but is unavoidable.
- **Stochastic results:** The dataset generator uses `torch.rand` without a fixed seed, so amplitudes `A` and phases `φ` differ each run. Qualitative conclusions (PINN > black-box; dummy data improves generalisation) are robust, but exact MSE values will vary.
- **Dummy data density trade-off:** Finer `X_dummy` grids enforce the physics more strongly but increase computation per epoch. The default step of 0.01 (`arange(-10, 10, 0.01)` = 2000 points) is a practical balance. For a step of 0.001, physics enforcement is stronger but each epoch is ~10× slower.
- **Quadrotor `gdown` links:** The dataset download links may expire or require re-authentication. If `gdown` fails, the files must be placed manually in the Colab session at `/content/train.csv` and `/content/test.csv`.
- **CosineAnnealingLR and local minima:** The scheduler is essential for PINNs because the physics and data losses can create conflicting gradient directions that trap the optimiser. The cosine restarts periodically increase the learning rate, helping escape such minima. Removing the scheduler often leads to premature convergence at suboptimal solutions.
- **dX_train scale ambiguity:** The quadrotor training code includes a runtime check `if abs(dX_test.mean()) < 0.1` to determine whether `dX_train` represents state changes `Δx` or derivatives `dx/dt`. This is needed because the same CSV data is loaded in different cells with different normalisation conventions. Always verify the scale of `dX_train` before interpreting loss values.

---

## Author

**Umer Ahmed Baig Mughal**
Master's in Robotics and Artificial Intelligence
Specialization: Machine Learning · Computer Vision · Human-Robot Interaction · Autonomous Systems · Robotic Motion Control

---

## License

This project is intended for academic and research use. It was developed as part of the Machine Learning for Robotics course within the MSc Robotics and Artificial Intelligence program at ITMO University. Redistribution, modification, and use in derivative academic work are permitted with appropriate attribution to the original author.

---

*Lab 2 — Machine Learning for Robotics | MSc Robotics and Artificial Intelligence | ITMO University*

