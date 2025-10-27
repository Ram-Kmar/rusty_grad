# Rusty Grad

`rusty_grad` is a foundational machine learning library written in Rust, inspired by frameworks like PyTorch. It aims to provide a flexible and efficient platform for building and training neural networks, featuring automatic differentiation and support for multiple computational backends.

## Core Features

*   **Tensor Operations**: A multi-dimensional `Tensor` struct as the core data structure.
*   **Automatic Differentiation**: (Planned) The architecture is designed to support gradient tracking for building computational graphs.
*   **Backend Agnostic**: Computation is abstracted through a `Backend` trait, allowing for different hardware accelerators.
    *   **CPU Backend**: A fully implemented backend for CPU-based computation.
    *   **CUDA Backend**: A backend for leveraging NVIDIA GPUs via CUDA for accelerated computations.
*   **Neural Network Primitives**:
    *   **Modules**: A `Module` trait for building neural network layers (e.g., `Linear`, `Embedding`).
    *   **Models**: A `Model` struct to chain layers together.
    *   **Optimizers**: An `Optimizer` trait with an initial `SGD` implementation.
    *   **Initializers**: Weight initialization strategies like `Zeros`, `Ones`, and `Xavier`.
*   **CUDA Integration**: Includes CUDA kernels (e.g., for vector addition) and Rust bindings to execute them.

## Project Structure

The project is organized into several modules, each responsible for a specific part of the library's functionality:

| File                  | Description                                                                                              |
| --------------------- | -------------------------------------------------------------------------------------------------------- |
| `Cargo.toml`          | Project manifest, contains dependencies like `matrixmultiply`, `rand`, and `num-traits`.                   |
| `src/main.rs`         | The main entry point of the binary crate, currently a placeholder.                                       |
| `src/tensor.rs`       | Defines the core `Tensor` data structure.                                                                |
| `src/backend.rs`      | The main `Backend` trait that abstracts computation.                                                     |
| `src/cpu_backend.rs`  | Implements the `Backend` trait for CPU operations.                                                       |
| `src/cuda_backend.rs` | Implements the `Backend` trait for CUDA operations.                                                      |
| `src/storage.rs`      | `Storage` trait and implementations (`CpuStorage`, `CudaStorage`) for managing tensor data.              |
| `src/module.rs`       | The `Module` trait, the basic building block for neural network layers.                                  |
| `src/linear.rs`       | A `Linear` (fully-connected) layer module.                                                               |
| `src/embedding.rs`    | An `Embedding` layer module.                                                                             |
| `src/model.rs`        | A `Model` struct that holds a sequence of layers.                                                        |
| `src/optimizer.rs`    | `Optimizer` trait and `SGD` implementation for model training.                                           |
| `src/initializers.rs` | Contains weight initialization strategies.                                                               |
| `src/traits.rs`       | Defines generic traits for tensor operations (`UnaryOp`, `BinaryOp`, `TernaryOp`).                       |
| `src/error.rs`        | Defines custom error types for the library.                                                              |
| `src/vector.cu`       | A CUDA kernel for element-wise vector addition.                                                          |
| `src/cuda_bindings.rs`| Auto-generated Rust bindings for the CUDA kernels.                                                       |

## Building

This is a standard Rust project. You can build it using Cargo:

```bash
cargo build
```

## Concepts

### Tensor

The `Tensor` is the central data structure in `rusty_grad`. It represents a multi-dimensional array and holds a reference to its `Storage`, shape, stride, and the computational `Device` it lives on.

### Backend and Storage

The `Backend` trait provides an abstraction over the computational device (CPU or CUDA). Each backend has an associated `Storage` type that manages how and where the data is stored (e.g., in a `Vec<f32>` for the CPU or a GPU memory pointer for CUDA).

### Modules and Models

Neural network layers are built by implementing the `Module` trait. `rusty_grad` provides standard layers like `Linear`. These modules can be composed into a `Model` to create a complete neural network.

## Current Status & Future Work

`rusty_grad` is currently in the early stages of development. Many features are planned or partially implemented.

*   **Implemented**:
    *   Basic Tensor and Storage abstractions.
    *   CPU and CUDA backend structure.
    *   `Linear` and `Embedding` layers.
    *   `SGD` optimizer stub.
*   **To-Do**:
    *   Implement backpropagation (`tensorbackprop.rs`).
    *   Complete the implementation of tensor operations (`tensorbinaryops.rs`, `gemm.rs`, etc.).
    *   Flesh out CUDA operations and improve memory safety.
    *   Add more layers, activation functions, and loss functions.
    *   Write comprehensive tests and usage examples.
