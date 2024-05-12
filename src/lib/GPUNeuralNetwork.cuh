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
        void runMiniBatch(std::vector<std::unique_ptr<std::vector<float> > >& inputData, std::vector<std::unique_ptr<std::vector<float> > >& trueLabels, std::vector<Matrix>& gradientCostWeight, std::vector<Matrix>& gradientCostBias);
        void runTrainingExample(std::unique_ptr<std::vector<float> >& exampleInputData, std::unique_ptr<std::vector<float> >& trueLabel, std::vector<Matrix>& gradientCostWeight, std::vector<Matrix>& gradientCostBias);

        void randomizeMiniBatches(std::vector<std::unique_ptr<std::vector<float> > >& allTrainingData, std::vector<std::vector<std::unique_ptr<std::vector<float> > > >& miniBatches, std::vector<std::unique_ptr<std::vector<float> > >& trueLabels, std::vector<std::vector<std::unique_ptr<std::vector<float> > > >& trueLabelsBatches, int miniBatchSize, std::default_random_engine& rng);

    public:
        GPUNeuralNetwork(std::string costFunc, int inputLayerNeurons, int numOutputClasses, float learningRate);
        ~GPUNeuralNetwork(); //destructor to free pointers in vector of layer*
        void initializeLayers(std::vector<std::string> layerTypes, std::vector<int> layerCounts);

        void trainNetwork(int numEpochs, std::vector<std::unique_ptr<std::vector<float> > >& allTrainingData, std::vector<std::unique_ptr<std::vector<float> > >& trueLabels, int miniBatchSize);

        std::vector<Layer*> layers; //default initialization is fine for this vector, no need to specify non-default initialization in constructor's member initializer list
        CostFunction costFunction;
        int numInputLayerNeurons;
        int numOutputClasses;
        float learningRate;
        Matrix inputActivations;
        Matrix trueOutput; //one hot vector encoding

};
__global__ void costWeightGradientExample(float* error, float* prev_a, float* g_w, int xDim, int yDim)
#endif