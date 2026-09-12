"""
Method C: Quantum-Inspired Optimization Algorithms
====================================================
Implements quantum-inspired optimization algorithms that run on classical
computers but borrow principles from quantum mechanics.

Based on:
- Superpositional Gradient Descent (IEEE QAI 2025)
- Quantum Hamiltonian Descent (ICML 2025)
- Quantum-inspired QUBO solvers
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Callable
from copy import deepcopy


class SuperpositionalGradientDescent(torch.optim.Optimizer):
    """
    Superpositional Gradient Descent (SGD-QI).
    
    A quantum-inspired optimizer that injects quantum circuit perturbations
    into gradient updates, enabling simultaneous exploration of multiple
    parameter configurations via quantum superposition principles.
    
    Reference: "Superpositional Gradient Descent" (IEEE QAI 2025)
    
    Key idea: Instead of following a single gradient direction,
    maintain a "superposition" of parameter states and combine their
    gradients for more robust optimization.
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        lambda_qi: float = 0.3,
        n_superpositions: int = 4,
        weight_decay: float = 0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta_1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta_2: {betas[1]}")
        
        defaults = dict(
            lr=lr, betas=betas, eps=eps,
            lambda_qi=lambda_qi,
            n_superpositions=n_superpositions,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step with quantum-inspired perturbations."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lam = group["lambda_qi"]
            n_sup = group["n_superpositions"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("SGD-QI does not support sparse gradients")
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    # Quantum-inspired: maintain superposition states
                    state["superpositions"] = [
                        p.data.clone() + torch.randn_like(p.data) * lam
                        for _ in range(n_sup)
                    ]
                    state["superposition_weights"] = torch.ones(n_sup) / n_sup
                
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                
                # Update biased first and second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                
                # Quantum-inspired perturbation: compute weighted superposition
                sup_grad = torch.zeros_like(grad)
                for i, sup_state in enumerate(state["superpositions"]):
                    # Compute gradient at superposition point (approximation)
                    sup_perturbation = (sup_state - p.data) * 0.1
                    sup_grad_i = grad + sup_perturbation
                    sup_grad += state["superposition_weights"][i] * sup_grad_i
                
                # Combine classical and quantum-inspired gradients
                combined_grad = (1 - lam) * grad + lam * sup_grad
                
                # Adam-like update with combined gradient
                denom = (exp_avg_sq / bias_correction2).sqrt().add_(eps)
                step_size = lr / bias_correction1
                
                p.data.add_(-step_size * combined_grad / denom)
                
                # Update superposition states
                for i in range(n_sup):
                    noise = torch.randn_like(p.data) * lam * 0.01
                    state["superpositions"][i] = p.data.clone() + noise
        
        return loss


class QuantumHamiltonianDescent:
    """
    Quantum Hamiltonian Descent (QHD) optimizer.
    
    Leverages quantum tunneling to escape saddle points and local minima.
    The classical simulation uses Hamiltonian dynamics with gradient information.
    
    Reference: "Quantum Optimization via Gradient-Based Hamiltonian Descent" (ICML 2025)
    
    Key insight: Quantum tunneling allows the optimizer to pass through
    energy barriers rather than climbing over them, potentially finding
    global minima faster than classical methods.
    """
    
    def __init__(
        self,
        params: List[torch.Tensor],
        lr: float = 0.01,
        mass: float = 1.0,
        damping: float = 0.1,
        dt: float = 0.01,
    ):
        self.params = params
        self.lr = lr
        self.mass = mass
        self.damping = damping
        self.dt = dt
        
        # Momentum (velocity) for each parameter
        self.velocities = [torch.zeros_like(p) for p in params]
    
    def step(self, grads: List[torch.Tensor]):
        """Perform one QHD step using Hamiltonian dynamics."""
        for i, (p, v, g) in enumerate(zip(self.params, self.velocities, grads)):
            # Hamiltonian dynamics: F = -grad V, with damping
            # dv/dt = -grad/m - gamma*v + quantum_tunneling
            # dp/dt = v
            
            # Quantum tunneling term: oscillatory perturbation
            tunneling = torch.randn_like(p) * self.lr * 0.1
            
            # Velocity update (with damping)
            self.velocities[i] = (
                v * (1 - self.damping * self.dt)
                - g / self.mass * self.dt
                + tunneling * self.dt
            )
            
            # Position update
            self.params[i] = p + self.velocities[i] * self.dt


class QuantumInspiredSimulatedAnnealing:
    """
    Quantum-Inspired Simulated Annealing (QISA).
    
    Combines classical simulated annealing with quantum-inspired
    tunneling moves for better exploration of the loss landscape.
    """
    
    def __init__(
        self,
        params: List[torch.Tensor],
        initial_temp: float = 1.0,
        cooling_rate: float = 0.99,
        tunneling_prob: float = 0.1,
    ):
        self.params = params
        self.temp = initial_temp
        self.cooling_rate = cooling_rate
        self.tunneling_prob = tunneling_prob
        self.best_params = [p.clone() for p in params]
        self.best_loss = float("inf")
    
    def step(self, loss: float, grads: List[torch.Tensor]):
        """Perform one QISA step."""
        # Update best
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_params = [p.clone() for p in self.params]
        
        for i, (p, g) in enumerate(zip(self.params, grads)):
            if np.random.random() < self.tunneling_prob:
                # Quantum tunneling: make a large jump
                tunnel_move = torch.randn_like(p) * self.temp * 2
                self.params[i] = p - self.lr * g + tunnel_move
            else:
                # Standard gradient descent with thermal noise
                thermal_noise = torch.randn_like(p) * self.temp * 0.1
                self.params[i] = p - self.lr * g + thermal_noise
        
        # Cool down
        self.temp *= self.cooling_rate
    
    def restore_best(self):
        """Restore the best parameters found."""
        for i, p in enumerate(self.params):
            p.data = self.best_params[i].clone()


class QuantumInspiredPSO:
    """
    Quantum-Inspired Particle Swarm Optimization (QI-PSO).
    
    Each particle maintains a superposition of positions,
    collapsing to a definite position when measured.
    """
    
    def __init__(
        self,
        n_particles: int = 20,
        param_shape: Tuple = (),
        lr: float = 0.01,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
    ):
        self.n_particles = n_particles
        self.lr = lr
        
        # Initialize particles
        self.positions = [
            torch.randn(*param_shape) * 0.1 for _ in range(n_particles)
        ]
        self.velocities = [
            torch.zeros(*param_shape) for _ in range(n_particles)
        ]
        self.personal_best_pos = [p.clone() for p in self.positions]
        self.personal_best_loss = [float("inf")] * n_particles
        
        self.global_best_pos = self.positions[0].clone()
        self.global_best_loss = float("inf")
        
        self.w, self.c1, self.c2 = w, c1, c2
    
    def update(self, losses: List[float]):
        """Update all particles."""
        for i in range(self.n_particles):
            if losses[i] < self.personal_best_loss[i]:
                self.personal_best_loss[i] = losses[i]
                self.personal_best_pos[i] = self.positions[i].clone()
            
            if losses[i] < self.global_best_loss:
                self.global_best_loss = losses[i]
                self.global_best_pos = self.positions[i].clone()
        
        for i in range(self.n_particles):
            r1, r2 = torch.rand_like(self.positions[i]), torch.rand_like(self.positions[i])
            
            # Update velocity
            self.velocities[i] = (
                self.w * self.velocities[i]
                + self.c1 * r1 * (self.personal_best_pos[i] - self.positions[i])
                + self.c2 * r2 * (self.global_best_pos - self.positions[i])
            )
            
            # Quantum-inspired: add superposition noise
            noise = torch.randn_like(self.positions[i]) * 0.01
            self.positions[i] += self.velocities[i] * self.lr + noise
    
    def get_best(self) -> torch.Tensor:
        return self.global_best_pos.clone()


def benchmark_optimizers(
    func: Callable,
    x_init: torch.Tensor,
    n_steps: int = 200,
) -> dict:
    """Benchmark different optimizers on a test function."""
    results = {}
    
    # Standard SGD
    x = x_init.clone().requires_grad_(True)
    opt = torch.optim.SGD([x], lr=0.01)
    losses_sgd = []
    for _ in range(n_steps):
        opt.zero_grad()
        loss = func(x)
        loss.backward()
        opt.step()
        losses_sgd.append(loss.item())
    results["SGD"] = losses_sgd
    
    # Adam
    x = x_init.clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=0.01)
    losses_adam = []
    for _ in range(n_steps):
        opt.zero_grad()
        loss = func(x)
        loss.backward()
        opt.step()
        losses_adam.append(loss.item())
    results["Adam"] = losses_adam
    
    # Quantum-Inspired SGD
    x = x_init.clone().requires_grad_(True)
    opt = SuperpositionalGradientDescent([x], lr=0.01, lambda_qi=0.3)
    losses_qi = []
    for _ in range(n_steps):
        opt.zero_grad()
        loss = func(x)
        loss.backward()
        opt.step()
        losses_qi.append(loss.item())
    results["SGD-QI"] = losses_qi
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Method C: Quantum-Inspired Optimization Algorithms")
    print("=" * 60)
    
    # Test function: Rastrigin (many local minima)
    def rastrigin(x):
        A = 10
        return A * len(x) + sum(
            xi**2 - A * torch.cos(2 * np.pi * xi) for xi in x
        )
    
    # Benchmark
    x_init = torch.randn(5) * 2
    print("\nBenchmarking on Rastrigin function (5D):")
    
    results = benchmark_optimizers(rastrigin, x_init, n_steps=300)
    
    for name, losses in results.items():
        print(f"  {name:10s}: final_loss = {losses[-1]:.4f}, min_loss = {min(losses):.4f}")
    
    # Test Superpositional GD on a simple classification
    print("\n[2] Superpositional GD on classification:")
    torch.manual_seed(42)
    X = torch.randn(50, 4)
    y = (X[:, 0] + X[:, 1] > 0).long()
    
    model = nn.Linear(4, 2)
    opt = SuperpositionalGradientDescent(model.parameters(), lr=0.01, lambda_qi=0.3)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    for epoch in range(100):
        opt.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    
    with torch.no_grad():
        acc = (model(X).argmax(1) == y).float().mean()
    print(f"    Final loss: {losses[-1]:.4f}")
    print(f"    Training accuracy: {acc:.2%}")
    
    print("\n" + "=" * 60)
    print("Method C demonstration complete.")
