#ifndef GPUNEURALNETWORK_H
#define GPUNEURALNETWORK_H

#include <iostream>
#include <vector>
#include <string>
#include "Layer.cuh"
#include "CostFunction.cuh"

class GPUNeuralNetwork {
    private:
        void runEpoch();
        void runMiniBatch();
        void runTrainingExample();

    public:
        GPUNeuralNetwork(std::string costFunc, int inputLayerNeurons, float learningRate);
        ~GPUNeuralNetwork(); //destructor to free pointers in vector of layer*
        void initializeLayers(std::vector<std::string> layerTypes, std::vector<int> layerCounts);

        void trainNetwork(int numEpochs, int numTrainingExamples, int miniBatchSize);

        std::vector<Layer*> layers;
        CostFunction costFunction;
        int numInputLayerNeurons;
        float* inputLayerActivations;
        float learningRate;

};
#endif