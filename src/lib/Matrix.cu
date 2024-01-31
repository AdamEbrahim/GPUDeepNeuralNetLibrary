#include "Matrix.cuh"
#include <iostream>

Matrix::Matrix(int xDim, int yDim) 
                : xDim{xDim}, yDim{yDim} { //constructor

}

void Matrix::allocateCUDAMemory() {
    float* getCudaMem = nullptr;
    cudaMalloc(&getCudaMem, xDim * yDim * sizeof(float));
    valuesDevice = std::shared_ptr<float>(getCudaMem, [&](float* ptr){ cudaFree(ptr); }); //must call cudaFree instead of free when destroying smart pointer
    std::cout << "Finished allocating device memory for matrix" << std::endl;
}

void Matrix::allocateHostMemory() {
    valuesHost = std::shared_ptr<float[]>(new float[xDim * yDim]);
    std::cout << "Finished allocating host memory for matrix" << std::endl;
}