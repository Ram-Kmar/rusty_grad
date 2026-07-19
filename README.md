# Rusty Grad

`rusty_grad` is a foundational machine learning library written in Rust, inspired by frameworks like PyTorch. It aims to provide a flexible and efficient platform for building and training neural networks, featuring automatic differentiation and support for multiple computational backends.

## Core Features

*   **Tensor Operations**: A multi-dimensional `Tensor` struct with support for basic arithmetic and matrix operations.
*   **Automatic Differentiation**: (Initial Implementation) Supports gradient tracking and backpropagation through a computational graph using topological sorting.
*   **Backend Agnostic**: Computation is abstracted through a `Backend` trait.
    *   **CPU Backend**: A functional backend for CPU-based computation utilizing `matrixmultiply` for high-performance GEMM.
    *   **CUDA Backend**: (Skeleton) A planned backend for leveraging NVIDIA GPUs; currently exists as an architectural placeholder.
*   **Neural Network Primitives**:
    *   **Modules**: A `Module` trait for building neural network layers (WIP).
    *   **Optimizers**: Functional `SGD` implementation for model training.
    *   **Initializers**: Weight initialization strategies like `Uniform`, `Zeros`, and `Ones`.

## Project Structure

The project is organized into several distinct modules:

| Module              | Description                                                                                              |
| ------------------- | -------------------------------------------------------------------------------------------------------- |
| `src/tensor/`       | Defines the core `Tensor` structures, operations (`unary`, `binary`, `movement`), and backpropagation.   |
| `src/backends/`     | Contains hardware-specific computation implementations (e.g., `cpu`, `cuda`).                            |
| `src/nn/`           | High-level neural network components including models, layers, optimizers (e.g., SGD), and initializers. |
| `src/core/`         | Foundational types and traits (e.g., `Device`, `Storage`, `TensorError`).                                |
| `src/math/`         | Core mathematical utilities like high-performance GEMM operations.                                       |
| `src/utils/`        | General helper functions and utilities.                                                                  |
| `src/main.rs`       | Demonstrates a working training loop with a basic multi-layer perceptron architecture.                   |

## Current Status & Future Work

`rusty_grad` is in active early development. While the core engine is functional, many high-level features are experimental.

### Implemented
*   **Tensor & Storage**: Basic abstractions with `RefCell` and `Shared` (Arc/Rc) for memory management.
*   **Autograd Engine**: Topological sorting and backward pass for `add`, `matmul`, `sigmoid`, `relu`, `tanh`, `exp`, `log`, etc.
*   **CPU Backend**: Efficient matrix multiplication via `matrixmultiply`.
*   **Training Loop**: Ability to perform forward/backward passes and update weights via SGD.

### Work in Progress (WIP)
*   **Layers**: `Linear` and `Embedding` layers are partially implemented and require stabilization.
*   **CUDA Integration**: Integration with a CUDA driver (like `cudarc`) is required to make the backend functional.
*   **Broadcasting**: Full tensor broadcasting support for binary operations.

### To-Do
*   Implement more complex layers (Conv2d, LayerNorm).
*   Add more loss functions (CrossEntropy, MSE).
*   Flesh out the CUDA backend for GPU acceleration.
*   Improve memory safety and performance in the autograd graph.
*   Comprehensive unit testing and documentation.

## Building

This is a standard Rust project:

```bash
cargo build
```

You can run the demonstration training loop in `main.rs` using:

```bash
cargo run
```
