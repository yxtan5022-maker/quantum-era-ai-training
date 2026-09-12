"""
Unified Experiment Runner
==========================
Runs all experiments across the four methods and generates comparative results.
"""

import sys
import os
import time
import json
import numpy as np
import torch

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_experiment_A():
    """Method A: Quantum Neural Network training."""
    from src.method_a_quantum_hardware.quantum_neural_network import (
        QuantumNeuralNetwork, generate_classification_data
    )
    
    print("\n" + "=" * 60)
    print("EXPERIMENT A: Quantum Neural Network Training")
    print("=" * 60)
    
    results = {}
    
    for n_qubits in [2, 4]:
        for n_layers in [2, 3]:
            X_train, y_train, X_test, y_test = generate_classification_data(
                n_samples=80, n_features=n_qubits, n_classes=2
            )
            
            model = QuantumNeuralNetwork(
                n_qubits=n_qubits, n_layers=n_layers, n_classes=2
            )
            
            start = time.time()
            losses = model.train_standard(X_train, y_train, epochs=60)
            train_time = time.time() - start
            
            acc = model.evaluate(X_test, y_test)
            
            key = f"qubit{n_qubits}_layer{n_layers}"
            results[key] = {
                "accuracy": acc,
                "final_loss": losses[-1],
                "train_time": train_time,
            }
            print(f"  {key}: acc={acc:.2%}, loss={losses[-1]:.4f}, time={train_time:.2f}s")
    
    return results


def run_experiment_B():
    """Method B: Hybrid Quantum-Classical models."""
    from src.method_b_hybrid.hybrid_model import (
        HybridQuantumClassicalModel, generate_hybrid_data
    )
    
    print("\n" + "=" * 60)
    print("EXPERIMENT B: Hybrid Quantum-Classical Training")
    print("=" * 60)
    
    results = {}
    X_train, y_train, X_test, y_test = generate_hybrid_data(
        n_samples=120, input_dim=4, n_classes=2
    )
    
    # Classical baseline
    import torch.nn as nn
    classical_model = nn.Sequential(
        nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 2)
    )
    opt = torch.optim.Adam(classical_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    start = time.time()
    for _ in range(60):
        opt.zero_grad()
        loss = criterion(classical_model(X_train), y_train)
        loss.backward()
        opt.step()
    classical_time = time.time() - start
    classical_acc = (classical_model(X_test).argmax(1) == y_test).float().mean().item()
    
    results["classical"] = {"accuracy": classical_acc, "train_time": classical_time}
    print(f"  Classical baseline: acc={classical_acc:.2%}, time={classical_time:.2f}s")
    
    # Hybrid model
    for n_qubits in [2, 4]:
        model = HybridQuantumClassicalModel(
            input_dim=4, n_qubits=n_qubits, n_quantum_layers=2,
            hidden_dim=32, n_classes=2
        )
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        
        start = time.time()
        for _ in range(60):
            opt.zero_grad()
            loss = criterion(model(X_train), y_train)
            loss.backward()
            opt.step()
        hybrid_time = time.time() - start
        hybrid_acc = (model(X_test).argmax(1) == y_test).float().mean().item()
        
        key = f"hybrid_q{n_qubits}"
        results[key] = {"accuracy": hybrid_acc, "train_time": hybrid_time}
        print(f"  Hybrid (q{n_qubits}): acc={hybrid_acc:.2%}, time={hybrid_time:.2f}s")
    
    return results


def run_experiment_C():
    """Method C: Quantum-Inspired Optimizers."""
    from src.method_c_quantum_inspired.quantum_inspired_optimizers import (
        SuperpositionalGradientDescent, benchmark_optimizers
    )
    
    print("\n" + "=" * 60)
    print("EXPERIMENT C: Quantum-Inspired Optimization")
    print("=" * 60)
    
    # Test on Rastrigin function
    def rastrigin(x):
        A = 10
        return A * len(x) + sum(xi**2 - A * torch.cos(2 * np.pi * xi) for xi in x)
    
    x_init = torch.randn(5) * 2
    results = benchmark_optimizers(rastrigin, x_init, n_steps=300)
    
    output = {}
    for name, losses in results.items():
        output[name] = {
            "final_loss": losses[-1],
            "min_loss": min(losses),
            "convergence_speed": sum(1 for i in range(1, len(losses)) if losses[i] < losses[i-1]),
        }
        print(f"  {name:10s}: final={losses[-1]:.4f}, min={min(losses):.4f}")
    
    # Classification comparison
    print("\n  Classification benchmark:")
    torch.manual_seed(42)
    X = torch.randn(50, 4)
    y = (X[:, 0] + X[:, 1] > 0).long()
    
    import torch.nn as nn
    results_cls = {}
    
    for opt_name, OptClass, kwargs in [
        ("Adam", torch.optim.Adam, {"lr": 0.01}),
        ("SGD-QI", SuperpositionalGradientDescent, {"lr": 0.01, "lambda_qi": 0.3}),
    ]:
        model = nn.Linear(4, 2)
        opt = OptClass(model.parameters(), **kwargs)
        criterion = nn.CrossEntropyLoss()
        
        start = time.time()
        for _ in range(100):
            opt.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            opt.step()
        elapsed = time.time() - start
        acc = (model(X).argmax(1) == y).float().mean().item()
        
        results_cls[opt_name] = {"accuracy": acc, "time": elapsed}
        print(f"    {opt_name}: acc={acc:.2%}, time={elapsed:.2f}s")
    
    output["classification"] = results_cls
    return output


def run_experiment_D():
    """Method D: Theoretical analysis summary."""
    from src.method_d_theoretical.quantum_advantage_analysis import (
        QuantumAdvantageAnalyzer
    )
    
    print("\n" + "=" * 60)
    print("EXPERIMENT D: Theoretical Analysis")
    print("=" * 60)
    
    analyzer = QuantumAdvantageAnalyzer()
    analyzer.analyze_gradient_estimation()
    analyzer.analyze_matrix_operations()
    analyzer.analyze_training_convergence()
    analyzer.analyze_neural_network_inference()
    analyzer.analyze_reinforcement_learning()
    analyzer.analyze_kernel_methods()
    
    results = {
        "n_components": len(analyzer.results),
        "exponential_speedups": sum(1 for r in analyzer.results if r.speedup_type == "exponential"),
        "polynomial_speedups": sum(1 for r in analyzer.results if r.speedup_type == "polynomial"),
        "quadratic_speedups": sum(1 for r in analyzer.results if r.speedup_type == "quadratic"),
    }
    
    print(f"  Components analyzed: {results['n_components']}")
    print(f"  Exponential speedups: {results['exponential_speedups']}")
    print(f"  Polynomial speedups: {results['polynomial_speedups']}")
    print(f"  Quadratic speedups: {results['quadratic_speedups']}")
    
    return results


def main():
    print("=" * 60)
    print("QUANTUM-ERA AI TRAINING RESEARCH")
    print("Unified Experiment Suite")
    print("=" * 60)
    
    all_results = {}
    
    all_results["method_A"] = run_experiment_A()
    all_results["method_B"] = run_experiment_B()
    all_results["method_C"] = run_experiment_C()
    all_results["method_D"] = run_experiment_D()
    
    # Save results
    results_path = os.path.join(os.path.dirname(__file__), "experiment_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("Results saved to: experiments/experiment_results.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
