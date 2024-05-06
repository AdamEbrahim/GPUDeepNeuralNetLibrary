#ifndef GPUNEURALNETWORK_H
#define GPUNEURALNETWORK_H

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include "Layer.cuh"
#include "CostFunction.cuh"

class GPUNeuralNetwork {
    private:
        void runMiniBatch(std::vector<std::unique_ptr<std::vector<float> > >& inputData);
        void runTrainingExample();

        void randomizeMiniBatches(std::vector<std::unique_ptr<std::vector<float> > >& allTrainingData, std::vector<std::vector<std::unique_ptr<std::vector<float> > > >& miniBatches, int miniBatchSize, std::default_random_engine& rng);

    public:
        GPUNeuralNetwork(std::string costFunc, int inputLayerNeurons, float learningRate);
        ~GPUNeuralNetwork(); //destructor to free pointers in vector of layer*
        void initializeLayers(std::vector<std::string> layerTypes, std::vector<int> layerCounts);

        void trainNetwork(int numEpochs, std::vector<std::unique_ptr<std::vector<float> > >& allTrainingData, int miniBatchSize);

        std::vector<Layer*> layers;
        CostFunction costFunction;
        int numInputLayerNeurons;
        float* inputLayerActivations;
        float learningRate;

};
#endif