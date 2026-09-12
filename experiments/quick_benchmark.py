"""
Quick benchmark: single-run results for paper tables.
"""

import sys, os, time, json
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    results = {}
    
    # === Method A: QNN ===
    print("[A] Quantum Neural Network...")
    from src.method_a_quantum_hardware.quantum_neural_network import (
        QuantumNeuralNetwork, generate_classification_data
    )
    
    for n_q in [2, 4]:
        for n_l in [2, 3]:
            torch.manual_seed(42)
            np.random.seed(42)
            X_tr, y_tr, X_te, y_te = generate_classification_data(80, n_q, 2)
            m = QuantumNeuralNetwork(n_qubits=n_q, n_layers=n_l, n_classes=2)
            t0 = time.time()
            losses = m.train_standard(X_tr, y_tr, epochs=60)
            dt = time.time() - t0
            acc = m.evaluate(X_te, y_te)
            key = f"A_q{n_q}l{n_l}"
            results[key] = {"accuracy": round(acc, 4), "loss": round(losses[-1], 4), "time": round(dt, 2)}
            print(f"  {key}: acc={acc:.2%} loss={losses[-1]:.4f} time={dt:.1f}s")
    
    # === Method B: Hybrid ===
    print("[B] Hybrid Quantum-Classical...")
    from src.method_b_hybrid.hybrid_model import (
        HybridQuantumClassicalModel, generate_hybrid_data
    )
    
    # Classical baseline
    torch.manual_seed(42)
    X_tr, y_tr, X_te, y_te = generate_hybrid_data(120, 4, 2)
    cm = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 2))
    opt = torch.optim.Adam(cm.parameters(), lr=0.01)
    crit = nn.CrossEntropyLoss()
    t0 = time.time()
    for _ in range(60):
        opt.zero_grad(); crit(cm(X_tr), y_tr).backward(); opt.step()
    dt = time.time() - t0
    acc = (cm(X_te).argmax(1) == y_te).float().mean().item()
    results["B_classical"] = {"accuracy": round(acc, 4), "loss": round(crit(cm(X_tr), y_tr).item(), 4), "time": round(dt, 2)}
    print(f"  Classical baseline: acc={acc:.2%} time={dt:.1f}s")
    
    for n_q in [2, 4]:
        torch.manual_seed(42)
        np.random.seed(42)
        X_tr, y_tr, X_te, y_te = generate_hybrid_data(120, 4, 2)
        hm = HybridQuantumClassicalModel(4, n_q, 2, 32, 2)
        opt = torch.optim.Adam(hm.parameters(), lr=0.01)
        t0 = time.time()
        for _ in range(60):
            opt.zero_grad(); crit(hm(X_tr), y_tr).backward(); opt.step()
        dt = time.time() - t0
        acc = (hm(X_te).argmax(1) == y_te).float().mean().item()
        key = f"B_hybrid_q{n_q}"
        results[key] = {"accuracy": round(acc, 4), "loss": round(crit(hm(X_tr), y_tr).item(), 4), "time": round(dt, 2)}
        print(f"  {key}: acc={acc:.2%} time={dt:.1f}s")
    
    # === Method C: Optimizers ===
    print("[C] Quantum-Inspired Optimizers...")
    from src.method_c_quantum_inspired.quantum_inspired_optimizers import SuperpositionalGradientDescent
    
    torch.manual_seed(42)
    X = torch.randn(60, 4)
    y = (X[:, 0] + X[:, 1] > 0).long()
    X_tr, X_te = X[:48], X[48:]
    y_tr, y_te = y[:48], y[48:]
    
    for opt_name, OptCls, kwargs in [
        ("Adam", torch.optim.Adam, {"lr": 0.01}),
        ("SGD-QI", SuperpositionalGradientDescent, {"lr": 0.01, "lambda_qi": 0.3}),
    ]:
        torch.manual_seed(42)
        m = nn.Linear(4, 2)
        o = OptCls(m.parameters(), **kwargs)
        crit = nn.CrossEntropyLoss()
        t0 = time.time()
        for _ in range(100):
            o.zero_grad(); crit(m(X_tr), y_tr).backward(); o.step()
        dt = time.time() - t0
        acc = (m(X_te).argmax(1) == y_te).float().mean().item()
        key = f"C_{opt_name}"
        results[key] = {"accuracy": round(acc, 4), "loss": round(crit(m(X_tr), y_tr).item(), 4), "time": round(dt, 2)}
        print(f"  {key}: acc={acc:.2%} time={dt:.1f}s")
    
    # === Method D: Theoretical ===
    print("[D] Theoretical Analysis...")
    from src.method_d_theoretical.quantum_advantage_analysis import QuantumAdvantageAnalyzer
    analyzer = QuantumAdvantageAnalyzer()
    analyzer.analyze_gradient_estimation()
    analyzer.analyze_matrix_operations()
    analyzer.analyze_training_convergence()
    analyzer.analyze_neural_network_inference()
    analyzer.analyze_reinforcement_learning()
    analyzer.analyze_kernel_methods()
    results["D_analysis"] = {
        "n_components": len(analyzer.results),
        "exponential_speedups": sum(1 for r in analyzer.results if r.speedup_type == "exponential"),
    }
    print(f"  Analyzed {len(analyzer.results)} components, {results['D_analysis']['exponential_speedups']} exponential speedups")
    
    # Save
    path = os.path.join(os.path.dirname(__file__), "final_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    # Print final table
    print("\n" + "=" * 65)
    print(f"{'Method':<20} {'Accuracy':>10} {'Loss':>10} {'Time (s)':>10}")
    print("=" * 65)
    for k, v in results.items():
        if "accuracy" in v:
            print(f"{k:<20} {v['accuracy']:>10.2%} {v['loss']:>10.4f} {v['time']:>10.1f}")
    print("=" * 65)
    print("Done!")


if __name__ == "__main__":
    main()
