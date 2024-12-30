# CUDA Project: Monte Carlo Pricing and Neural Network Training for Asian Options

## Project Overview
This project implements CUDA kernels to efficiently price Asian options using Monte Carlo simulations and trains neural networks on the sampled data to approximate the price function.

The **Black-Scholes model** is used under the risk-neutral measure for spot price dynamics. Simulations leverage parallel processing on GPUs using CUDA for high computational efficiency.

---

## Key Features
- **Monte Carlo Simulations**:
  - Parallel generation of sample paths for the Black-Scholes stochastic differential equation using the Euler–Maruyama scheme.
  - Efficient computation of Asian option prices via Monte Carlo estimation.

- **Neural Network Training**:
  - Training a Multi-Layer Perceptron (MLP) to approximate the pricing function.
  - Support for direct training on payoffs or Monte Carlo price estimations.
  - Regularization techniques, including Layer Normalization and ReLU activations.

- **GPU Acceleration**:
  - CUDA kernels for parallel path generation.
  - Optimized array reduction for aggregating simulation results.

---

## Mathematical Formulation

1. **Black-Scholes Model**  
   $$
   dS_t = r\,S_t\,dt + \sigma\,S_t\,dW(t).
   $$

   **Discretized using Euler–Maruyama**:  
   $$
   S_{t_{k+1}} = S_{t_k}\biggl(1 + r\,\frac{T}{n} 
   + \sigma\,\sqrt{\tfrac{T}{n}}\;Z_{k+1}\biggr),
   $$  
   where \(Z_{k+1}\) are i.i.d. standard normal variables.

2. **Asian Option Pricing**  
   $$F(t, T, S_t, I_t, r, \sigma) = e^{-r(T-t)} 
   \,\mathbb{E}\bigl[\,(S_T - I_T)^+ \mid S_t, I_t\bigr],$$  
   with  
   $$I_t = \frac{1}{t}\,\int_{0}^{t} S_u \,du.$$

3. **Neural Network Loss**  
   We want to learn the price function \(F(x)\). The training objective is:
   $$\theta^* \;=\; \underset{\theta \in \Theta}{\arg\min}\
   \mathbb{E}_{x \sim D} \Bigl[\bigl(F(x) - T_\theta(x)\bigr)^2\Bigr].$$

---

## Implementation Details

1. **Monte Carlo Simulation**:
   - Parallelized over trajectories via CUDA threads.
   - Optimized for GPU using data reduction kernels.

2. **Neural Network Architecture**:
   - Fully connected MLP with 4 hidden layers, each containing 400 neurons.
   - Activation function: SiLU.
   - Regularization: LayerNorm between hidden layers, and ReLU to ensure nonnegative outputs.

3. **Training and Data Generation**:
   - **Training**:
     - Generate \(10^6\) parameter samples, each with \(10^3\) paths using a scrambled Halton sequence.
   - **Validation**:
     - Generate \(10^4\) parameter samples, each with \(10^6\) paths for a Monte Carlo estimation of \(F\).

---

## Prerequisites
- CUDA-compatible GPU
- Python 3.x
- Python dependencies (Numba, PyTorch, etc.) – see [`requirements.txt`](./requirements.txt)

---

## Installation and Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/cuda-project.git
   cd cuda-project
