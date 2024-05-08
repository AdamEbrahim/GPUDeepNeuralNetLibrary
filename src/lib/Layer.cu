#include "Layer.cuh"
#include <random>
#include <cmath>

Layer::Layer(int prevNumNeurons, int numNeurons) : weights{prevNumNeurons, numNeurons}, biases{1, numNeurons}, weightedInput{1, numNeurons}, outputActivation{1, numNeurons}, inputError{1, numNeurons}, numberNeurons{numNeurons}, prevLayerNumNeurons{prevNumNeurons} {
    weights.allocateHostMemory();
    weights.allocateCUDAMemory();

    biases.allocateHostMemory();
    biases.allocateCUDAMemory();

    weightedInput.allocateHostMemory();
    weightedInput.allocateCUDAMemory();

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

    cudaMemcpy(biases.valuesDevice.get(), biases.valuesHost.get(), biases.xDim * biases.yDim * sizeof(float), cudaMemcpyHostToDevice);

    //initialize weights to be random float
    //use a normal gaussian distribution, mean = 0, std_dev = 1 / sqrt(input_neurons). Weights can be negative.
    std::default_random_engine generator;
    std::normal_distribution<float> dist(0, 1.0 / sqrt(prevLayerNumNeurons)); 
    for (int i = 0; i < weights.xDim * weights.yDim; i++) {
        weights.valuesHost[i] = dist(generator);
    }

    cudaMemcpy(weights.valuesDevice.get(), weights.valuesHost.get(), weights.xDim * weights.yDim * sizeof(float), cudaMemcpyHostToDevice);

}

void Layer::forwardPass(Matrix& prevLayerActivations) {
    float* x = prevLayerActivations.valuesDevice.get(); //no need to cudaMemcpy, updated values will already be on device
    float* w = (this->weights).valuesDevice.get();
    float* b = (this->biases).valuesDevice.get();
    float* z = (this->weightedInput).valuesDevice.get();
    float* a = (this->outputActivation).valuesDevice.get();

    //figure out block/grid dimensions:
    int num_threads = 256; //just set 256 threads per block now; testing to do
    int num_blocks = std::ceil((1.0 * this->weights.yDim) / num_threads);
    dim3 blocks(num_blocks);
    dim3 threads(num_threads);
    
    callGetActivation(blocks, threads, w, x, a, b, z, this->weights.xDim, this->weights.yDim);
    cudaDeviceSynchronize();
    cudaMemcpy(this->outputActivation.valuesHost.get(), this->outputActivation.valuesDevice.get(), this->outputActivation.yDim * sizeof(float), cudaMemcpyDeviceToHost);
}

void Layer::backprop(Matrix& nextLayerError, Matrix& nextLayerWeights) {
    float* nextError = nextLayerError.valuesDevice.get();
    float* w = nextLayerWeights.valuesDevice.get();
    float* z = (this->weightedInput).valuesDevice.get();
    float* error = (this->inputError).valuesDevice.get();

    //figure out block/grid dimensions:
    int num_threads = 256; //just set 256 threads per block now; testing to do
    int num_blocks = std::ceil((1.0 * this->weights.yDim) / num_threads);
    dim3 blocks(num_blocks);
    dim3 threads(num_threads);

    callBackPropError(blocks, threads, nextError, w, z, error, nextLayerWeights.xDim, nextLayerWeights.yDim);
    cudaDeviceSynchronize();
    cudaMemcpy(this->inputError.valuesHost.get(), this->inputError.valuesDevice.get(), this->inputError.yDim * sizeof(float), cudaMemcpyDeviceToHost);
}