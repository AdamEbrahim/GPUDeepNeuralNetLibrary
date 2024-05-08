#include "GPUNeuralNetwork.cuh"
#include "SigmoidLayer.cuh"
#include "Matrix.cuh"
#include <algorithm>
#include <random>
#include <numeric>

GPUNeuralNetwork::GPUNeuralNetwork(std::string costFunc, int inputLayerNeurons, float learningRate) try : costFunction(costFunc), numInputLayerNeurons{inputLayerNeurons}, learningRate{learningRate}, inputActivations{1, inputLayerNeurons} 
    {
        //allocate host and cuda memory for input layer activations
        inputActivations.allocateHostMemory();
        inputActivations.allocateCUDAMemory();
    }
    catch (std::string type) {
        std::cout << "Invalid cost function: " << type << std::endl;
    }

GPUNeuralNetwork::~GPUNeuralNetwork() { //destructor
    for (int i = 0; i < this->layers.size(); i++) {
        delete this->layers[i];
    }
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
void GPUNeuralNetwork::runTrainingExample(std::unique_ptr<std::vector<float> >& exampleInputData, std::vector<Matrix>& gradientCostWeight, std::vector<Matrix>& gradientCostBias) { 
    //set input layer activations
    uint32_t len = this->inputActivations.xDim * this->inputActivations.yDim;
    if (len != (*exampleInputData).size()) {
        std::cout << "error: improperly sized training input" << std::endl;
    }

    for (int i = 0; i < len; i++) {
        this->inputActivations.valuesHost[i] = (*exampleInputData)[i];
    }
    cudaMemcpy(this->inputActivations.valuesDevice.get(), this->inputActivations.valuesHost.get(), this->inputActivations.xDim * this->inputActivations.yDim * sizeof(float), cudaMemcpyHostToDevice);

    //forward pass through each layer of network
    for (int i = 0; i < this->layers.size(); i++) {
        if (i == 0) {
            this->layers[i]->forwardPass(this->inputActivations);
        } else {
            this->layers[i]->forwardPass(this->layers[i-1]->outputActivation);
        }
    }

    //obtain cost/loss of current training input and use it to compute error of the final layer

    //backpropagate error through each layer of network

    //Update gradient matrices
}

//Mini batch will call runTrainingExample() on all training inputs in mini batch of size m, use that to perform gradient descent
//inputData is a vector of size m, where each element inputData[m] is a pointer to a vector of one training example's input layer encodings 
void GPUNeuralNetwork::runMiniBatch(std::vector<std::unique_ptr<std::vector<float> > >& inputData, std::vector<Matrix>& gradientCostWeight, std::vector<Matrix>& gradientCostBias) {
    int miniBatchSize = inputData.size();
    if (miniBatchSize == 0) {
        return;
    }

    //reinit running sum of gradients for each layer to 0
    for (int i = 0; i < this->layers.size(); i++) {
        Matrix& w = gradientCostWeight[i];
        Matrix& b = gradientCostBias[i];

        for (int j = 0; j < w.xDim * w.yDim; j++) {
            w.valuesHost[j] = 0.0;
        }

        for (int j = 0; j < b.xDim * b.yDim; j++) {
            b.valuesHost[j] = 0.0;
        }

        cudaMemcpy(w.valuesDevice.get(), w.valuesHost.get(), w.xDim * w.yDim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(b.valuesDevice.get(), b.valuesHost.get(), b.xDim * b.yDim * sizeof(float), cudaMemcpyHostToDevice);
    }

    
    for (int i = 0; i < miniBatchSize; i++) {
        //std::cout << "[" << (*(inputData[i]))[0] << ", " << (*(inputData[i]))[1] << ", " << (*(inputData[i]))[2] << "]" << std::endl;
        runTrainingExample(inputData[i], gradientCostWeight, gradientCostBias); //will update weight and bias gradients
    }

    //obtain average of the gradients after running all training inputs
    //float avgWeightGradient = std::accumulate(gradientCostWeight.begin(), gradientCostWeight.end(), 0.0) / miniBatchSize;
    //float avgBiasGradient = std::accumulate(gradientCostBias.begin(), gradientCostBias.end(), 0.0) / miniBatchSize;

    //Update the weights and biases
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

    auto rng = std::default_random_engine {std::random_device {}()}; //create a reusable instance of default random engine, the () is function call operator overloading after instantiating the random device seed

    //Matrices for each layer of network, keep a running sum of total gradients for each weight and bias in the layer as you go through each training data. Will be averaged after.
    std::vector<Matrix> gradientCostWeight;
    std::vector<Matrix> gradientCostBias;
    for (int i = 0; i < this->layers.size(); i++) {
        gradientCostWeight.emplace_back(Matrix((this->layers[i])->weights.xDim, (this->layers[i])->weights.yDim));
        gradientCostBias.emplace_back(Matrix((this->layers[i])->biases.xDim, (this->layers[i])->biases.yDim));

        Matrix& w = gradientCostWeight[i];
        Matrix& b = gradientCostBias[i];

        w.allocateHostMemory();
        w.allocateCUDAMemory();
        b.allocateHostMemory();
        b.allocateCUDAMemory();
    }
    
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
            runMiniBatch(miniBatches[j], gradientCostWeight, gradientCostBias);
        }
        
    }

}