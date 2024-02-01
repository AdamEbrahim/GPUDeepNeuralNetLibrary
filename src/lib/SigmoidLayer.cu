#include "SigmoidLayer.cuh"

//Kernel functions to be run on GPU
__global__ void SigmoidLayer::getActivation(float* x) {

}


SigmoidLayer::SigmoidLayer(int prevNumNeurons, int numNeurons) : Layer{prevNumNeurons, numNeurons} {

}

void SigmoidLayer::forwardPass(Matrix& prevLayerActivations) {
    float* prev = prevLayerActivations.valuesDevice.get(); //no need to cudaMemcpy, updated values will already be on device
    
}