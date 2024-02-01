#include <iostream>
#include <vector>
#include <string>
#include "Layer.cuh"
#include "CostFunction.cuh"

class GPUNeuralNetwork {
    private:

    public:
        GPUNeuralNetwork(std::string costFunc, int inputLayerNeurons, float learningRate);
        void initializeLayers(std::vector<std::string> layerTypes, std::vector<int> layerCounts);

        std::vector<Layer*> layers;
        CostFunction costFunction;
        int inputLayerNeurons;
        float learningRate;

};