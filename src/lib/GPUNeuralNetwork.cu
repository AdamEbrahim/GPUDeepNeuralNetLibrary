#include "GPUNeuralNetwork.cuh"
#include "SigmoidLayer.cuh"
#include <algorithm>
#include <random>

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
    std::cout << "hi" << std::endl;
}

//Mini batch will call runTrainingExample() on all training inputs in mini batch of size m, use that to perform gradient descent
//inputData is a vector of size m, where each element inputData[m] is a vector of one training example's input layer encodings 
void GPUNeuralNetwork::runMiniBatch(std::vector<std::unique_ptr<std::vector<float> > >& inputData) {
    int miniBatchSize = inputData.size();
    for (int i = 0; i < inputData.size(); i++) {
        std::cout << "[" << (*(inputData[i]))[0] << ", " << (*(inputData[i]))[1] << ", " << (*(inputData[i]))[2] << "]" << std::endl;
    }
}

void GPUNeuralNetwork::randomizeMiniBatches(std::vector<std::unique_ptr<std::vector<float> > >& allTrainingData, std::vector<std::vector<std::unique_ptr<std::vector<float> > > >& miniBatches, int miniBatchSize, std::default_random_engine& rng) {
    std::shuffle(allTrainingData.begin(), allTrainingData.end(), rng);

    for (int i = 0; i < allTrainingData.size(); i++) {
        int currBatchIdx = i / miniBatchSize;
        miniBatches[currBatchIdx].push_back(std::move(allTrainingData[i]));
    }
    //clear all the remaining nullptrs after move
    allTrainingData.clear();
}

void GPUNeuralNetwork::trainNetwork(int numEpochs, std::vector<std::unique_ptr<std::vector<float> > >& allTrainingData, int miniBatchSize) {
    int numMiniBatches = (allTrainingData.size() / miniBatchSize) + 1;
    std::vector<std::vector<std::unique_ptr<std::vector<float> > > > miniBatches;
    for (int i = 0; i < numMiniBatches; i++) {
        miniBatches.push_back(std::vector<std::unique_ptr<std::vector<float> > >(0));
    }
    auto rd = std::random_device {}; //randomly seed the random generator used for vector shuffle
    auto rng = std::default_random_engine {rd()}; //create a reusable instance of default random engine, rd() is function call operator overloading

    for (int i = 0; i < numEpochs; i++) {
        std::cout << "Beginning Epoch " << i << " of training:" << std::endl;

        //Move everything from mini batch vectors back to allTrainingData vector
        if (i != 0) {
            for (int j = 0; j < numMiniBatches; j++) {
                for (int k = 0; k < miniBatches[j].size(); k++) {
                    allTrainingData.push_back(std::move(miniBatches[j][k]));
                }
                //clear all the remaining nullptrs after move
                miniBatches[j].clear();
            }

        }

        randomizeMiniBatches(allTrainingData, miniBatches, miniBatchSize, rng);
        for (int j = 0; j < numMiniBatches; j++) {
            std::cout << "Running Mini Batch " << j << std::endl;
            runMiniBatch(miniBatches[j]);
        }
        
    }

}