#include "GPUNeuralNetwork.cuh"
#include "SigmoidLayer.cuh"

GPUNeuralNetwork::GPUNeuralNetwork(std::string costFunc, int inputLayerNeurons, float learningRate) try : costFunction(costFunc), numInputLayerNeurons{inputLayerNeurons}, learningRate{learningRate} 
    {

    }
    catch (std::string type) {
        std::cout << "Invalid cost function: " << type << std::endl;
    }

GPUNeuralNetwork::~GPUNeuralNetwork() { //destructor
    for (int i = 0; i < this->layers.size(); i++) {
        delete this->layers[i];
    }
    delete[] inputLayerActivations; 
}

void GPUNeuralNetwork::initializeLayers(std::vector<std::string> layerTypes, std::vector<int> layerCounts) {
    try {
        if (layerTypes.size() != layerCounts.size()) {
            throw(layerTypes.size());
        }

        int prevLayerNeurons = 0;
        for (int i = 0; i < layerTypes.size(); i++) {
            
            if (i == 0) {
                prevLayerNeurons = this->numInputLayerNeurons;
            } else {
                prevLayerNeurons = layerCounts[i-1];
            }

            Layer* createdLayer;
            if (layerTypes[i] == "Sigmoid") {
                createdLayer = new SigmoidLayer(prevLayerNeurons, layerCounts[i]);
            } else {
                throw(layerTypes[i]);
            }

            createdLayer->initializeMatrices(); //initialize the matrices for layer
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

//single training example, will change input layer activations array to be proper values
void GPUNeuralNetwork::runTrainingExample() { 

}

//Mini batch will call runTrainingExample() on all training inputs in mini batch of size m, use that to perform gradient descent
void GPUNeuralNetwork::runMiniBatch() {

}

void GPUNeuralNetwork::runEpoch() { 

}

void GPUNeuralNetwork::trainNetwork(int numEpochs, int numTrainingExamples, int miniBatchSize) {

}