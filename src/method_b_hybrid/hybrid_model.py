"""
Method B: Hybrid Quantum-Classical Training Architecture
=========================================================
Implements a hybrid model where quantum circuits serve as trainable
feature transformation layers within a classical neural network.

Based on:
- Hybrid QCQ-CNN (Nature Scientific Reports, 2025)
- Quantum kernel methods (Rodriguez-Grasa et al., 2025)
- Scalable On-Hardware Training (Mathur et al., 2026)
"""

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


def create_quantum_layer(n_qubits: int = 4):
    """Create a PennyLane quantum layer for PyTorch integration."""
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def quantum_layer(inputs, weights):
        """Quantum feature transformation layer."""
        # Encode classical features
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        
        # Variational processing
        for i in range(weights.shape[0]):
            for j in range(n_qubits):
                qml.RY(weights[i, j], wires=j)
                qml.RZ(weights[i, j] * 0.5, wires=j)
            for j in range(n_qubits - 1):
                qml.CNOT(wires=[j, j + 1])
            # Entangle last with first (ring topology)
            qml.CNOT(wires=[n_qubits - 1, 0])
        
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    return quantum_layer


class HybridQuantumClassicalModel(nn.Module):
    """
    Hybrid Quantum-Classical Neural Network.
    
    Architecture:
    Classical Input → Classical Hidden → Quantum Layer → Classical Output
    
    The quantum layer provides a trainable feature transformation
    that operates in high-dimensional Hilbert space.
    """
    
    def __init__(
        self,
        input_dim: int = 4,
        n_qubits: int = 4,
        n_quantum_layers: int = 2,
        hidden_dim: int = 64,
        n_classes: int = 2,
    ):
        super().__init__()
        
        self.n_qubits = n_qubits
        
        # Classical pre-processing
        self.pre_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_qubits),
            nn.Tanh(),  # Bound to [-1, 1] for angle embedding
        )
        
        # Quantum layer (as torch function)
        self.quantum_weights = nn.Parameter(
            torch.randn(n_quantum_layers, n_qubits) * 0.1
        )
        
        # Classical post-processing
        self.post_layer = nn.Sequential(
            nn.Linear(n_qubits, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes),
        )
        
        self._quantum_func = create_quantum_layer(n_qubits)
    
    def quantum_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply quantum transformation to a single sample."""
        q_out = self._quantum_func(x, self.quantum_weights)
        return torch.stack([r if isinstance(r, torch.Tensor) else torch.tensor(r) for r in q_out])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: classical → quantum → classical."""
        # Pre-process
        h = self.pre_layer(x)
        
        # Quantum transformation (batch processing)
        quantum_outputs = []
        for i in range(h.shape[0]):
            q_out = self.quantum_transform(h[i])
            quantum_outputs.append(q_out)
        quantum_features = torch.stack(quantum_outputs).float()
        
        # Post-process
        logits = self.post_layer(quantum_features)
        return logits


class QuantumKernelSVM:
    """
    Quantum Kernel Methods for classification.
    
    Uses quantum circuits to compute kernel matrices
    in exponentially large Hilbert spaces.
    """
    
    def __init__(self, n_qubits: int = 4, n_layers: int = 2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(self.dev, interface="numpy")
        def kernel_circuit(x1, x2, weights):
            """Compute quantum kernel between two data points."""
            # Encode first data point
            qml.AngleEmbedding(x1, wires=range(n_qubits), rotation="Y")
            # Apply variational layers
            for i in range(n_layers):
                for j in range(n_qubits):
                    qml.RY(weights[i, j], wires=j)
                for j in range(n_qubits - 1):
                    qml.CNOT(wires=[j, j + 1])
            # Inverse of second encoding
            qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits), rotation="Y")
            
            return qml.expval(qml.PauliZ(0))
        
        self.kernel_circuit = kernel_circuit
        self.weights = np.random.randn(n_layers, n_qubits) * 0.1
    
    def kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute the full kernel matrix."""
        n1, n2 = X1.shape[0], X2.shape[0]
        K = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self.kernel_circuit(X1[i], X2[j], self.weights)
        
        return K
    
    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 50):
        """Train the quantum kernel parameters."""
        # Simple gradient-free optimization for demonstration
        from scipy.optimize import minimize
        
        def objective(params):
            params_reshaped = params.reshape(self.n_layers, self.n_qubits)
            self.weights = params_reshaped
            K = self.kernel_matrix(X, X)
            # Simple hinge loss on kernel predictions
            y_pred = np.sign(K @ y)
            loss = np.mean(np.maximum(0, 1 - y * y_pred))
            return loss
        
        result = minimize(
            objective,
            self.weights.flatten(),
            method="COBYLA",
            options={"maxiter": epochs},
        )
        self.weights = result.x.reshape(self.n_layers, self.n_qubits)
    
    def predict(self, X: np.ndarray, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Predict using trained kernel."""
        K_test = self.kernel_matrix(X, X_train)
        y_pred = np.sign(K_test @ y_train)
        return y_pred


class QuantumConvLayer(nn.Module):
    """
    Quantum Convolutional Layer (QCNN).
    
    Applies quantum convolution on local patches of the input,
    followed by pooling to reduce dimensionality.
    """
    
    def __init__(self, n_qubits: int = 4):
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.conv_weights = nn.Parameter(torch.randn(n_qubits) * 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply quantum convolution."""
        # Simplified: apply parameterized rotation to each qubit
        # In practice, would implement full QCNN with entangling layers
        output = []
        for i in range(x.shape[0]):
            @qml.qnode(self.dev, interface="torch", diff_method="backprop")
            def conv_circuit(inputs, weights):
                for j in range(self.n_qubits):
                    qml.RY(inputs[j % len(inputs)], wires=j)
                    qml.RY(weights[j], wires=j)
                for j in range(self.n_qubits - 1):
                    qml.CNOT(wires=[j, j + 1])
                return [qml.expval(qml.PauliZ(j)) for j in range(self.n_qubits)]
            
            q_out = conv_circuit(x[i], self.conv_weights)
            output.append(torch.stack([r if isinstance(r, torch.Tensor) else torch.tensor(r) for r in q_out]))
        
        return torch.stack(output)


def generate_hybrid_data(
    n_samples: int = 200, input_dim: int = 4, n_classes: int = 2
) -> Tuple:
    """Generate data for hybrid model benchmarking."""
    from sklearn.datasets import make_moons
    
    X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
    
    # Pad to input_dim if needed
    if X.shape[1] < input_dim:
        X = np.hstack([X, np.zeros((n_samples, input_dim - X.shape[1]))])
    X = X[:, :input_dim]
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Method B: Hybrid Quantum-Classical Training Architecture")
    print("=" * 60)
    
    # Test Hybrid Model
    print("\n[1] Hybrid Quantum-Classical Model:")
    X_train, y_train, X_test, y_test = generate_hybrid_data(
        n_samples=100, input_dim=4, n_classes=2
    )
    
    model = HybridQuantumClassicalModel(
        input_dim=4, n_qubits=4, n_quantum_layers=2, hidden_dim=32, n_classes=2
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    for epoch in range(60):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    with torch.no_grad():
        test_output = model(X_test)
        acc = (test_output.argmax(1) == y_test).float().mean()
    
    print(f"    Final loss: {losses[-1]:.4f}")
    print(f"    Test accuracy: {acc:.2%}")
    
    # Test Quantum Kernel SVM
    print("\n[2] Quantum Kernel SVM:")
    X_train_np = X_train.numpy()
    y_train_np = np.where(y_train.numpy() == 0, -1, 1).astype(float)
    X_test_np = X_test.numpy()
    y_test_np = np.where(y_test.numpy() == 0, -1, 1).astype(float)
    
    qsvm = QuantumKernelSVM(n_qubits=4, n_layers=2)
    qsvm.fit(X_train_np, y_train_np, epochs=30)
    y_pred = qsvm.predict(X_test_np, X_train_np, y_train_np)
    acc_kernel = np.mean(y_pred == y_test_np)
    print(f"    Test accuracy: {acc_kernel:.2%}")
    
    print("\n" + "=" * 60)
    print("Method B demonstration complete.")
