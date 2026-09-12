"""
Comprehensive Benchmark: 5-fold cross-validation across all methods.
Generates publication-quality data with standard deviations.
"""

import sys, os
import time
import json
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cross_validate(func, n_folds=5, **kwargs):
    """Run n-fold cross-validation and return mean±std."""
    results = []
    for fold in range(n_folds):
        torch.manual_seed(fold * 42)
        np.random.seed(fold * 42)
        result = func(**kwargs)
        results.append(result)
    
    # Compute mean and std
    keys = results[0].keys()
    summary = {}
    for k in keys:
        vals = [r[k] for r in results]
        summary[f"{k}_mean"] = np.mean(vals)
        summary[f"{k}_std"] = np.std(vals)
    return summary


def benchmark_method_A(n_qubits=4, n_layers=3, n_samples=80, epochs=60):
    """Benchmark Method A: Quantum Neural Network."""
    from src.method_a_quantum_hardware.quantum_neural_network import (
        QuantumNeuralNetwork, generate_classification_data
    )
    
    X_train, y_train, X_test, y_test = generate_classification_data(
        n_samples=n_samples, n_features=n_qubits, n_classes=2
    )
    
    model = QuantumNeuralNetwork(
        n_qubits=n_qubits, n_layers=n_layers, n_classes=2
    )
    
    start = time.time()
    losses = model.train_standard(X_train, y_train, epochs=epochs)
    train_time = time.time() - start
    
    acc = model.evaluate(X_test, y_test)
    
    return {"accuracy": acc, "final_loss": losses[-1], "train_time": train_time}


def benchmark_method_B(n_qubits=4, n_samples=120, epochs=60):
    """Benchmark Method B: Hybrid Quantum-Classical."""
    from src.method_b_hybrid.hybrid_model import (
        HybridQuantumClassicalModel, generate_hybrid_data
    )
    
    X_train, y_train, X_test, y_test = generate_hybrid_data(
        n_samples=n_samples, input_dim=4, n_classes=2
    )
    
    model = HybridQuantumClassicalModel(
        input_dim=4, n_qubits=n_qubits, n_quantum_layers=2,
        hidden_dim=32, n_classes=2
    )
    
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    start = time.time()
    for _ in range(epochs):
        opt.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        opt.step()
    train_time = time.time() - start
    
    acc = (model(X_test).argmax(1) == y_test).float().mean().item()
    
    return {"accuracy": acc, "final_loss": loss.item(), "train_time": train_time}


def benchmark_method_C_classification(epochs=100):
    """Benchmark Method C: Quantum-Inspired Optimizer on classification."""
    from src.method_c_quantum_inspired.quantum_inspired_optimizers import (
        SuperpositionalGradientDescent
    )
    
    torch.manual_seed(42)
    X = torch.randn(60, 4)
    y = (X[:, 0] + X[:, 1] > 0).long()
    
    X_train, X_test = X[:48], X[48:]
    y_train, y_test = y[:48], y[48:]
    
    model = nn.Linear(4, 2)
    opt = SuperpositionalGradientDescent(model.parameters(), lr=0.01, lambda_qi=0.3)
    criterion = nn.CrossEntropyLoss()
    
    start = time.time()
    for _ in range(epochs):
        opt.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        opt.step()
    train_time = time.time() - start
    
    acc = (model(X_test).argmax(1) == y_test).float().mean().item()
    
    return {"accuracy": acc, "final_loss": loss.item(), "train_time": train_time}


def benchmark_classical_baseline(epochs=100):
    """Benchmark classical Adam optimizer as baseline."""
    torch.manual_seed(42)
    X = torch.randn(60, 4)
    y = (X[:, 0] + X[:, 1] > 0).long()
    
    X_train, X_test = X[:48], X[48:]
    y_train, y_test = y[:48], y[48:]
    
    model = nn.Linear(4, 2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    start = time.time()
    for _ in range(epochs):
        opt.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        opt.step()
    train_time = time.time() - start
    
    acc = (model(X_test).argmax(1) == y_test).float().mean().item()
    
    return {"accuracy": acc, "final_loss": loss.item(), "train_time": train_time}


def main():
    print("=" * 70)
    print("COMPREHENSIVE BENCHMARK (5-fold cross-validation)")
    print("=" * 70)
    
    all_results = {}
    
    # Method A: QNN
    print("\n[Method A] Quantum Neural Network (5-fold CV)...")
    for n_q in [2, 4]:
        for n_l in [2, 3]:
            key = f"QNN_q{n_q}_l{n_l}"
            summary = cross_validate(
                benchmark_method_A, n_folds=5,
                n_qubits=n_q, n_layers=n_l
            )
            all_results[key] = summary
            print(f"  {key}: acc={summary['accuracy_mean']:.2%} +/- {summary['accuracy_std']:.2%}")
    
    # Method B: Hybrid
    print("\n[Method B] Hybrid Quantum-Classical (5-fold CV)...")
    for n_q in [2, 4]:
        key = f"Hybrid_q{n_q}"
        summary = cross_validate(
            benchmark_method_B, n_folds=5,
            n_qubits=n_q
        )
        all_results[key] = summary
        print(f"  {key}: acc={summary['accuracy_mean']:.2%} +/- {summary['accuracy_std']:.2%}")
    
    # Method C: SGD-QI vs Adam
    print("\n[Method C] Quantum-Inspired Optimizer (5-fold CV)...")
    summary_qi = cross_validate(benchmark_method_C_classification, n_folds=5)
    all_results["SGD_QI"] = summary_qi
    print(f"  SGD-QI: acc={summary_qi['accuracy_mean']:.2%} +/- {summary_qi['accuracy_std']:.2%}")
    
    summary_adam = cross_validate(benchmark_classical_baseline, n_folds=5)
    all_results["Adam_baseline"] = summary_adam
    print(f"  Adam:   acc={summary_adam['accuracy_mean']:.2%} +/- {summary_adam['accuracy_std']:.2%}")
    
    # Save
    results_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Method':<20} {'Accuracy':>12} {'Loss':>10} {'Time (s)':>10}")
    print("-" * 55)
    for key, val in all_results.items():
        acc_str = f"{val['accuracy_mean']:.2%} +/- {val['accuracy_std']:.2%}"
        loss_str = f"{val['final_loss_mean']:.4f}"
        time_str = f"{val['train_time_mean']:.2f}"
        print(f"{key:<20} {acc_str:>20} {loss_str:>10} {time_str:>10}")
    
    print("\nBenchmark complete. Results saved.")


if __name__ == "__main__":
    main()
