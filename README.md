# GPU Accelerated Deep Neural Network Library
This is an object-oriented software library that I wrote from scratch in C++ and CUDA for building and training GPU Accelerated Deep Neural Networks with various types of layer activations and loss functions.
CUDA is used to parallelize matrix/vector calculations on a compatible GPU (no external CUDA libraries beyond Nvidia's CUDA API are used), and optimizations such as memory coalescing, shared memory cache-blocking, and block tiling are implemented to achieve a matrix multiplication speed 85% of cuBLAS library.
