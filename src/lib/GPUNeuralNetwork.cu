#include "GPUNeuralNetwork.cuh"
#include "SigmoidLayer.cuh"

GPUNeuralNetwork::GPUNeuralNetwork(std::string costFunc, int inputLayerNeurons, float learningRate) try : costFunction(costFunc), inputLayerNeurons{inputLayerNeurons}, learningRate{learningRate} 
    {

    }
    catch (std::string type) {
        std::cout << "Invalid cost function: " << type << std::endl;
    }

void GPUNeuralNetwork::initializeLayers(std::vector<std::string> layerTypes, std::vector<int> layerCounts) {
    try {
        if (layerTypes.size() != layerCounts.size()) {
            throw(layerTypes.size());
        }

        int prevLayerNeurons = 0;
        for (int i = 0; i < layerTypes.size(); i++) {
            
            if (i == 0) {
                prevLayerNeurons = this->inputLayerNeurons;
            } else {
                prevLayerNeurons = layerCounts[i-1];
            }

            Layer* createdLayer;
            if (layerTypes[i] == "Sigmoid") {
                createdLayer = new SigmoidLayer(prevLayerNeurons, layerCounts[i]);
            } else {
                throw(layerTypes[i]);
            }
            layers.push_back(createdLayer);
        }
    }
    catch (int layerTypesSize) {
        std::cout << "Improperly sized initialize layers vectors" << std::endl;
    }
    catch (std::string layerName) {
        std::cout << "Improperly named layer type: " << layerName << std::endl;
    }

}