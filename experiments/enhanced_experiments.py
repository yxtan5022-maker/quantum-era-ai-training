"""
Enhanced experiments: larger datasets, ablation studies, noise robustness.
Generates publication-quality data.
"""

import sys, os, time, json
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons, make_circles

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_data(dataset, n_samples, noise=0.1):
    if dataset == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    elif dataset == "circles":
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
    else:
        X = np.random.randn(n_samples, 4)
        y = ((X[:, 0] + X[:, 1]) > 0).astype(int)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    idx = torch.randperm(n_samples)
    split = int(0.8 * n_samples)
    return X[idx[:split]], y[idx[:split]], X[idx[split:]], y[idx[split:]]


class ClassicalMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_classes)
        )
    def forward(self, x):
        return self.net(x)


def train_and_eval(model, X_tr, y_tr, X_te, y_te, lr=0.01, epochs=100, optimizer_cls=None):
    if optimizer_cls is None:
        optimizer_cls = torch.optim.Adam
    opt = optimizer_cls(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    
    t0 = time.time()
    for _ in range(epochs):
        opt.zero_grad()
        loss = crit(model(X_tr), y_tr)
        loss.backward()
        opt.step()
    dt = time.time() - t0
    
    acc = (model(X_te).argmax(1) == y_te).float().mean().item()
    final_loss = crit(model(X_te), y_te).item()
    return {"accuracy": acc, "loss": final_loss, "time": round(dt, 2)}


def experiment_classical_scaling():
    """Test how classical MLP scales with hidden dim."""
    print("\n[Exp 1] Classical MLP Scaling")
    X_tr, y_tr, X_te, y_te = make_data("moons", 200)
    results = {}
    for h in [8, 16, 32]:
        torch.manual_seed(42)
        m = ClassicalMLP(2, h, 2)
        r = train_and_eval(m, X_tr, y_tr, X_te, y_te, epochs=80)
        results[f"MLP_h{h}"] = r
        print(f"  h={h}: acc={r['accuracy']:.2%} time={r['time']:.1f}s")
    return results


def experiment_quantum_scaling():
    """Test QNN scaling with qubit count on larger dataset."""
    print("\n[Exp 2] QNN Scaling with Qubit Count")
    from src.method_a_quantum_hardware.quantum_neural_network import (
        QuantumNeuralNetwork, generate_classification_data
    )
    results = {}
    for n_q in [2, 3, 4]:
        torch.manual_seed(42)
        np.random.seed(42)
        X_tr, y_tr, X_te, y_te = generate_classification_data(80, n_q, 2)
        m = QuantumNeuralNetwork(n_qubits=n_q, n_layers=2, n_classes=2)
        r = train_and_eval(m, X_tr, y_tr, X_te, y_te, epochs=60)
        results[f"QNN_q{n_q}"] = r
        print(f"  q={n_q}: acc={r['accuracy']:.2%} loss={r['loss']:.4f} time={r['time']:.1f}s")
    return results


def experiment_hybrid_ablation():
    """Ablation: effect of quantum layer count in hybrid model."""
    print("\n[Exp 3] Hybrid Model Ablation (Quantum Layers)")
    from src.method_b_hybrid.hybrid_model import (
        HybridQuantumClassicalModel, generate_hybrid_data
    )
    results = {}
    for n_ql in [1, 2]:
        torch.manual_seed(42)
        np.random.seed(42)
        X_tr, y_tr, X_te, y_te = generate_hybrid_data(80, 4, 2)
        m = HybridQuantumClassicalModel(4, 4, n_ql, 32, 2)
        r = train_and_eval(m, X_tr, y_tr, X_te, y_te, epochs=40)
        results[f"Hybrid_ql{n_ql}"] = r
        print(f"  ql={n_ql}: acc={r['accuracy']:.2%} loss={r['loss']:.4f} time={r['time']:.1f}s")
    return results


def experiment_noise_robustness():
    """Test QNN robustness to input noise."""
    print("\n[Exp 4] Noise Robustness (QNN)")
    from src.method_a_quantum_hardware.quantum_neural_network import (
        QuantumNeuralNetwork, generate_classification_data
    )
    results = {}
    for noise_level in [0.0, 0.1, 0.2]:
        torch.manual_seed(42)
        np.random.seed(42)
        X_tr, y_tr, X_te, y_te = generate_classification_data(60, 2, 2)
        X_te_noisy = X_te + noise_level * torch.randn_like(X_te)
        m = QuantumNeuralNetwork(n_qubits=2, n_layers=2, n_classes=2)
        _ = train_and_eval(m, X_tr, y_tr, X_te, y_te, epochs=40)
        acc = (m(X_te_noisy).argmax(1) == y_te).float().mean().item()
        results[f"noise_{noise_level}"] = {"accuracy": acc}
        print(f"  noise={noise_level}: acc={acc:.2%}")
    return results


def experiment_optimizer_comparison():
    """Compare optimizers on non-convex landscape."""
    print("\n[Exp 5] Optimizer Comparison (Rastrigin 10D)")
    from src.method_c_quantum_inspired.quantum_inspired_optimizers import SuperpositionalGradientDescent
    
    def rastrigin(x):
        A = 10
        return A * x.shape[0] + torch.sum(x**2 - A * torch.cos(2 * np.pi * x))
    
    results = {}
    for opt_name, OptCls, kwargs in [
        ("SGD", torch.optim.SGD, {"lr": 0.01, "momentum": 0.9}),
        ("Adam", torch.optim.Adam, {"lr": 0.01}),
        ("SGD-QI", SuperpositionalGradientDescent, {"lr": 0.01, "lambda_qi": 0.3}),
    ]:
        torch.manual_seed(42)
        x = torch.randn(10, requires_grad=True)
        o = OptCls([x], **kwargs)
        t0 = time.time()
        for _ in range(200):
            o.zero_grad()
            loss = rastrigin(x)
            loss.backward()
            o.step()
        dt = time.time() - t0
        final = rastrigin(x).item()
        results[opt_name] = {"final_loss": round(final, 4), "time": round(dt, 2)}
        print(f"  {opt_name}: final={final:.4f} time={dt:.2f}s")
    return results


def experiment_dataset_comparison():
    """Compare classical vs hybrid on different datasets."""
    print("\n[Exp 6] Dataset Comparison (Classical vs Hybrid)")
    from src.method_b_hybrid.hybrid_model import HybridQuantumClassicalModel
    
    results = {}
    for dataset in ["moons", "circles"]:
        X_tr, y_tr, X_te, y_te = make_data(dataset, 100)
        
        # Classical
        torch.manual_seed(42)
        cm = ClassicalMLP(X_tr.shape[1], 32, 2)
        r_cl = train_and_eval(cm, X_tr, y_tr, X_te, y_te, epochs=60)
        
        # Hybrid
        torch.manual_seed(42)
        np.random.seed(42)
        hm = HybridQuantumClassicalModel(X_tr.shape[1], 2, 2, 32, 2)
        r_hy = train_and_eval(hm, X_tr, y_tr, X_te, y_te, epochs=60)
        
        results[dataset] = {"classical": r_cl, "hybrid": r_hy}
        print(f"  {dataset}: classical={r_cl['accuracy']:.2%} hybrid={r_hy['accuracy']:.2%}")
    return results


def main():
    print("=" * 70)
    print("ENHANCED EXPERIMENTS")
    print("=" * 70)
    
    all_results = {}
    all_results["classical_scaling"] = experiment_classical_scaling()
    all_results["quantum_scaling"] = experiment_quantum_scaling()
    all_results["hybrid_ablation"] = experiment_hybrid_ablation()
    all_results["noise_robustness"] = experiment_noise_robustness()
    all_results["optimizer_comparison"] = experiment_optimizer_comparison()
    all_results["dataset_comparison"] = experiment_dataset_comparison()
    
    path = os.path.join(os.path.dirname(__file__), "enhanced_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("All enhanced experiments complete. Results saved.")
    print("=" * 70)


if __name__ == "__main__":
    main()
