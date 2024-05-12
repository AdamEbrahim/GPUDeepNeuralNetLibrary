#include "CostFunction.cuh"
#include <iostream>

CostFunction::CostFunction(std::string type) {
    if (type == "BCE") {
        this->type = type;
    } else { //error checking for invalid cost function type
        throw(type);
    }
}

CostFunction::CostFunction() { //default constructor, default to BCE cost function
    this->type = "BCE";
}

__global__ void getErrorFinalLayerBCE(float* error, float* a, float* z, float* y, float* g_b, int xDim, int yDim) {
    int rowIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = rowIndex; i < yDim; i = i + stride) {
       error[i] = (((1.0 - y[i])/(1 - a[i])) - (y[i]/a[i])) * z[i];
       g_b[i] += error[i]; //gradient bias update for the training input = error
    }
}



void CostFunction::getErrorFinalLayer(Layer* finalLayer, Matrix& trueLabel, Matrix& gradientCostBias) {
    float* finalLayerInputError = finalLayer->inputError.valuesDevice.get();
    //Layer::act* activationPtr = finalLayer->activation.get();
    //Layer::act* activationPrimePtr = finalLayer->activationPrime.get();
    float* a = finalLayer->outputActivation.valuesDevice.get();
    float* z = finalLayer->outputActivationPrime.valuesDevice.get();
    float* y = trueLabel.valuesDevice.get();
    float* g_b = gradientCostBias.valuesDevice.get();

    //figure out block/grid dimensions:
    int num_threads = 256; //just set 256 threads per block now; testing to do
    int num_blocks = std::ceil((1.0 * trueLabel.yDim) / num_threads);
    dim3 blocks(num_blocks);
    dim3 threads(num_threads);


    if (this->type == "BCE") {
        getErrorFinalLayerBCE<<<blocks, threads>>>(finalLayerInputError, a, z, y, g_b, trueLabel.xDim, trueLabel.yDim);
    } else {
        std::cout << "Final layer error can't be computed due to unsupported loss function type" << std::endl;
    }

    cudaDeviceSynchronize();
    cudaMemcpy(finalLayer->inputError.valuesHost.get(), finalLayer->inputError.valuesDevice.get(), finalLayer->inputError.yDim * sizeof(float), cudaMemcpyDeviceToHost);
}

double CostFunction::getCostBCE() {
    double val = 1.0;

    return val;
}

double CostFunction::getCost() {
    if (this->type == "BCE") {
        return this->getCostBCE();
    } else {
        return 0.0;
    }
}