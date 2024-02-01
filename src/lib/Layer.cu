#include "Layer.cuh"
#include <random>
#include <cmath>

Layer::Layer(int prevNumNeurons, int numNeurons) : weights{prevNumNeurons, numNeurons}, biases{1, numNeurons}, outputActivation{1, numNeurons}, inputError{1, numNeurons}, numberNeurons{numNeurons}, prevLayerNumNeurons{prevNumNeurons} {
    weights.allocateHostMemory();
    weights.allocateCUDAMemory();

    biases.allocateHostMemory();
    biases.allocateCUDAMemory();

    outputActivation.allocateHostMemory();
    outputActivation.allocateCUDAMemory();

    inputError.allocateHostMemory();
    inputError.allocateCUDAMemory();
}

void Layer::initializeMatrices() {
    //initialize biases to 0
    for (int i = 0; i < biases.xDim * biases.yDim; i++) {
        biases.valuesHost[i] = 0.0;
    }

    cudaMemcpy(biases.valuesDevice, biases.valuesHost, biases.xDim * biases.yDim * sizeof(float), cudaMemcpyHostToDevice);

    //initialize weights to be random float [0-1]
    //use a normal gaussian distribution, mean = 0, std_dev = 1 / sqrt(input_neurons). Weights can be negative.
    std::default_random_engine generator;
    std::normal_distribution<float> dist(0, 1.0 / sqrt(prevLayerNumNeurons)); 
    for (int i = 0; i < weights.xDim * weights.yDim; i++) {
        weights.valuesHost[i] = dist(generator) * 0.1; //scale times 0.1 to get weight initialization between -1 and 1
    }

    cudaMemcpy(weights.valuesDevice, weights.valuesHost, weights.xDim * weights.yDim * sizeof(float), cudaMemcpyHostToDevice);

}