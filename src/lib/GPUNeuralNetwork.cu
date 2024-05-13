#include "GPUNeuralNetwork.cuh"
#include "SigmoidLayer.cuh"
#include "Matrix.cuh"
#include <algorithm>
#include <random>
#include <numeric>

GPUNeuralNetwork::GPUNeuralNetwork(std::string costFunc, int inputLayerNeurons, int numOutputClasses, float learningRate) try : costFunction(costFunc), numInputLayerNeurons{inputLayerNeurons}, numOutputClasses{numOutputClasses}, learningRate{learningRate}, inputActivations{1, inputLayerNeurons}, trueOutput{1, numOutputClasses} 
    {
        //allocate host and cuda memory for input layer activations
        inputActivations.allocateHostMemory();
        inputActivations.allocateCUDAMemory();
        //allocate host and cuda memory for true output vector
        trueOutput.allocateHostMemory();
        trueOutput.allocateCUDAMemory();
    }
    catch (std::string type) {
        std::cout << "Invalid cost function: " << type << std::endl;
    }

GPUNeuralNetwork::~GPUNeuralNetwork() { //destructor
    for (int i = 0; i < this->layers.size(); i++) {
        delete this->layers[i];
    }
}

//final layer count will no matter what be same as numOutputClasses, no matter what is given
void GPUNeuralNetwork::initializeLayers(std::vector<std::string>& layerTypes, std::vector<int>& layerCounts) {
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
            //final layer count will no matter what be same as numOutputClasses, no matter what is given
            int count = (i == layerTypes.size() - 1) ? this->numOutputClasses : layerCounts[i];
            if (layerTypes[i] == "Sigmoid") {
                createdLayer = new SigmoidLayer(prevLayerNeurons, count);
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

__global__ void costWeightGradientExample(float* error, float* prev_a, float* g_w, int xDim, int yDim) {
    int rowIndex = threadIdx.y + blockDim.y * blockIdx.y;
    int stride_y = blockDim.y * gridDim.y;
    int colIndex = threadIdx.x + blockDim.x + blockIdx.x;
    int stride_x = blockDim.x * gridDim.x;

    for (int i = rowIndex; i < yDim; i = i + stride_y) {
        for (int j = colIndex; j < xDim; j = j + stride_x) {
            g_w[(i * xDim) + j] += error[i] * prev_a[j];
        }
    }


}

__global__ void updateWeights(float* w, float* g_w, int m, float learningRate, int xDim, int yDim) {
    int rowIndex = threadIdx.y + blockDim.y * blockIdx.y;
    int stride_y = blockDim.y * gridDim.y;
    int colIndex = threadIdx.x + blockDim.x + blockIdx.x;
    int stride_x = blockDim.x * gridDim.x;

    for (int i = rowIndex; i < yDim; i = i + stride_y) {
        for (int j = colIndex; j < xDim; j = j + stride_x) {
            w[(i * xDim) + j] = w[(i * xDim) + j] - ((learningRate / (1.0 * m)) * g_w[(i * xDim) + j]);
        }
    }

}

__global__ void updateBiases(float* b, float* g_b, int m, float learningRate, int xDim, int yDim) {
    int rowIndex = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = rowIndex; i < yDim; i = i + stride) {
        b[i] = b[i] - ((learningRate / (1.0 * m)) * g_b[i]);
    }

}

//single training example, will change input layer activations array to be proper values
void GPUNeuralNetwork::runTrainingExample(std::unique_ptr<std::vector<float> >& exampleInputData, std::unique_ptr<std::vector<float> >& trueLabel, std::vector<Matrix>& gradientCostWeight, std::vector<Matrix>& gradientCostBias) { 
    //set input layer activations
    uint32_t len = this->inputActivations.xDim * this->inputActivations.yDim;
    if (len != (*exampleInputData).size()) {
        std::cout << "error: improperly sized training input" << std::endl;
    }

    for (int i = 0; i < len; i++) {
        this->inputActivations.valuesHost[i] = (*exampleInputData)[i];
    }
    cudaMemcpy(this->inputActivations.valuesDevice.get(), this->inputActivations.valuesHost.get(), this->inputActivations.xDim * this->inputActivations.yDim * sizeof(float), cudaMemcpyHostToDevice);

    //set true output label vector one hot encoded
    uint32_t len2 = this->trueOutput.xDim * this->trueOutput.yDim;
    if (len2 != (*trueLabel).size()) {
        std::cout << "error: improperly sized true label vector" << std::endl;
    }

    for (int i = 0; i < len2; i++) {
        this->trueOutput.valuesHost[i] = (*trueLabel)[i];
    }
    cudaMemcpy(this->trueOutput.valuesDevice.get(), this->trueOutput.valuesHost.get(), this->trueOutput.xDim * this->trueOutput.yDim * sizeof(float), cudaMemcpyHostToDevice);

    //forward pass through each layer of network
    for (int i = 0; i < this->layers.size(); i++) {
        if (i == 0) {
            this->layers[i]->forwardPass(this->inputActivations);
        } else {
            this->layers[i]->forwardPass(this->layers[i-1]->outputActivation);
        }
    }

    //backpropagate error through each layer of network to compute input error at each layer, update cost w.r.t bias gradient
    float* error;
    float* prev_a;
    float* g_w;

    int num_threadsx = 16; //just set 256 threads per block now; testing to do.
    int num_threadsy = 16;
    int num_blocksx;
    int num_blocksy;
    dim3 blocks;
    dim3 threads = dim3(num_threadsx, num_threadsy); //2d thread dimensions per block

    for (int i = this->layers.size() - 1; i >= 0; i--) {
        if (i == this->layers.size() - 1) { //looking at final layer
            //compute error of the final layer, update cost w.r.t bias gradient
            this->costFunction.getErrorFinalLayer(this->layers[i], this->trueOutput, gradientCostBias[i]);

        } else {
            //backprop error, update cost w.r.t bias gradient
            this->layers[i]->backprop(this->layers[i+1]->inputError, this->layers[i+1]->weights, gradientCostBias[i]);

        }

        //Update cost w.r.t weight gradient matrix
        error = this->layers[i]->inputError.valuesDevice.get();
        prev_a = (i == 0) ? this->inputActivations.valuesDevice.get() : this->layers[i-1]->outputActivation.valuesDevice.get();
        g_w = gradientCostWeight[i].valuesDevice.get();
        //figure out block/grid dimensions:
        num_blocksx = std::ceil((1.0 * gradientCostWeight[i].xDim) / num_threadsx);
        num_blocksy = std::ceil((1.0 * gradientCostWeight[i].yDim) / num_threadsy);
        blocks = dim3(num_blocksx, num_blocksy); //2d block dimensions in grid
        costWeightGradientExample<<<blocks, threads>>>(error, prev_a, g_w, gradientCostWeight[i].xDim, gradientCostWeight[i].yDim); //Update cost w.r.t weight gradient matrix
        cudaDeviceSynchronize(); 
        //no need to cudamemcpy to host until later
    }

}

//Mini batch will call runTrainingExample() on all training inputs in mini batch of size m, use that to perform gradient descent
//inputData is a vector of size m, where each element inputData[m] is a pointer to a vector of one training example's input layer encodings 
void GPUNeuralNetwork::runMiniBatch(std::vector<std::unique_ptr<std::vector<float> > >& inputData, std::vector<std::unique_ptr<std::vector<float> > >& trueLabels, std::vector<Matrix>& gradientCostWeight, std::vector<Matrix>& gradientCostBias) {
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
        runTrainingExample(inputData[i], trueLabels[i], gradientCostWeight, gradientCostBias); //will update weight and bias gradients
    }

    //obtain average of the gradients after running all training inputs and update all weights and biases
    int num_threadsx = 16; //just set 256 threads per block now; testing to do.
    int num_threadsy = 16;
    dim3 threads; //2d thread dimensions per block
    int num_blocksx;
    int num_blocksy;
    dim3 blocks; //2d block dimensions in grid

    for (int i = 0; i < this->layers.size(); i++) {
        num_threadsx = 16;
        num_blocksx = std::ceil((1.0 * gradientCostWeight[i].xDim) / num_threadsx);
        num_blocksy = std::ceil((1.0 * gradientCostWeight[i].yDim) / num_threadsy);
        threads = dim3(num_threadsx, num_threadsy);
        blocks = dim3(num_blocksx, num_blocksy);

        updateWeights<<<blocks, threads>>>(this->layers[i]->weights.valuesDevice.get(), gradientCostWeight[i].valuesDevice.get(), miniBatchSize, this->learningRate, gradientCostWeight[i].xDim, gradientCostWeight[i].yDim);
        cudaDeviceSynchronize(); 
        cudaMemcpy(this->layers[i]->weights.valuesHost.get(), this->layers[i]->weights.valuesDevice.get(), this->layers[i]->weights.xDim * this->layers[i]->weights.yDim * sizeof(float), cudaMemcpyDeviceToHost);

        num_threadsx = 256;
        num_blocksx = std::ceil((1.0 * gradientCostBias[i].yDim) / num_threadsx);
        threads = dim3(num_threadsx);
        blocks = dim3(num_blocksx);
        updateBiases<<<blocks, threads>>>(this->layers[i]->biases.valuesDevice.get(), gradientCostBias[i].valuesDevice.get(), miniBatchSize, this->learningRate, gradientCostBias[i].xDim, gradientCostBias[i].yDim);
        cudaDeviceSynchronize(); 
        cudaMemcpy(this->layers[i]->biases.valuesHost.get(), this->layers[i]->biases.valuesDevice.get(), this->layers[i]->biases.xDim * this->layers[i]->biases.yDim * sizeof(float), cudaMemcpyDeviceToHost);
    }

}

void GPUNeuralNetwork::randomizeMiniBatches(std::vector<std::unique_ptr<std::vector<float> > >& allTrainingData, std::vector<std::vector<std::unique_ptr<std::vector<float> > > >& miniBatches, std::vector<std::unique_ptr<std::vector<float> > >& trueLabels, std::vector<std::vector<std::unique_ptr<std::vector<float> > > >& trueLabelsBatches, int miniBatchSize, std::default_random_engine& rng) {
    std::vector<int> shuffleIndexes (allTrainingData.size(), 0);
    for (int i = 0; i < shuffleIndexes.size(); i++) {
        shuffleIndexes[i] = i;
    }
    std::shuffle(shuffleIndexes.begin(), shuffleIndexes.end(), rng);
    //std::shuffle(allTrainingData.begin(), allTrainingData.end(), rng);

    for (int i = 0; i < allTrainingData.size(); i++) {
        int currBatchIdx = i / miniBatchSize;
        miniBatches[currBatchIdx].push_back(std::move(allTrainingData[shuffleIndexes[i]]));
        trueLabelsBatches[currBatchIdx].push_back(std::move(trueLabels[shuffleIndexes[i]]));
    }
    //clear all the remaining nullptrs after move
    allTrainingData.clear();
    trueLabels.clear();
}

void GPUNeuralNetwork::trainNetwork(int numEpochs, std::vector<std::unique_ptr<std::vector<float> > >& allTrainingData, std::vector<std::unique_ptr<std::vector<float> > >& trueLabels, std::vector<std::unique_ptr<std::vector<float> > >& allTestingData, std::vector<std::unique_ptr<std::vector<float> > >& testingLabels, int miniBatchSize) {
    int numMiniBatches = (allTrainingData.size() / miniBatchSize) + 1;
    std::vector<std::vector<std::unique_ptr<std::vector<float> > > > miniBatches;
    std::vector<std::vector<std::unique_ptr<std::vector<float> > > > trueLabelsBatches;
    for (int i = 0; i < numMiniBatches; i++) {
        miniBatches.push_back(std::vector<std::unique_ptr<std::vector<float> > >(0));
        trueLabelsBatches.push_back(std::vector<std::unique_ptr<std::vector<float> > >(0));
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
                    trueLabels.push_back(std::move(trueLabelsBatches[j][k]));
                }
                //clear all the remaining nullptrs after move
                miniBatches[j].clear();
                trueLabelsBatches[j].clear();
            }

        }

        randomizeMiniBatches(allTrainingData, miniBatches, trueLabels, trueLabelsBatches, miniBatchSize, rng);
        for (int j = 0; j < numMiniBatches; j++) {
            //std::cout << "Running Mini Batch " << j << std::endl;
            runMiniBatch(miniBatches[j], trueLabelsBatches[j], gradientCostWeight, gradientCostBias);
        }

        //At end of epoch, get testing accuracy
        this->testNetwork(allTestingData, testingLabels);
        
    }

}

void GPUNeuralNetwork::testNetwork(std::vector<std::unique_ptr<std::vector<float> > >& testingData, std::vector<std::unique_ptr<std::vector<float> > >& trueLabels) {
    std::cout << "Testing Network..." << std::endl;

    int totalCorrect = 0;
    for (int j = 0; j < testingData.size(); j++) {
        //set input layer activations
        uint32_t len = this->inputActivations.xDim * this->inputActivations.yDim;
        if (len != (*(testingData[j])).size()) {
            std::cout << "error: improperly sized testing input" << std::endl;
        }

        for (int i = 0; i < len; i++) {
            this->inputActivations.valuesHost[i] = (*(testingData[j]))[i];
        }
        cudaMemcpy(this->inputActivations.valuesDevice.get(), this->inputActivations.valuesHost.get(), this->inputActivations.xDim * this->inputActivations.yDim * sizeof(float), cudaMemcpyHostToDevice);

        //set true output label vector one hot encoded
        uint32_t len2 = this->trueOutput.xDim * this->trueOutput.yDim;
        if (len2 != (*(trueLabels[j])).size()) {
            std::cout << "error: improperly sized true label vector" << std::endl;
        }

        for (int i = 0; i < len2; i++) {
            this->trueOutput.valuesHost[i] = (*(trueLabels[j]))[i];
        }
        cudaMemcpy(this->trueOutput.valuesDevice.get(), this->trueOutput.valuesHost.get(), this->trueOutput.xDim * this->trueOutput.yDim * sizeof(float), cudaMemcpyHostToDevice);

        //forward pass through each layer of network
        for (int i = 0; i < this->layers.size(); i++) {
            if (i == 0) {
                this->layers[i]->forwardPass(this->inputActivations);
            } else {
                this->layers[i]->forwardPass(this->layers[i-1]->outputActivation);
            }
        }

        //final layer output activations is predicted class
        //std::cout << "TESTING EXAMPLE #" << j << std::endl;
        float* predictedValues = this->layers[this->layers.size() - 1]->outputActivation.valuesHost.get();
        std::vector<float>& actualValues = *(trueLabels[j]);

        int predictedLabel = 0;
        float predictedLabelValue = -1.0;
        for (int i = 0; i < actualValues.size(); i++) {
            if (predictedValues[i] > predictedLabelValue) {
                predictedLabelValue = predictedValues[i];
                predictedLabel = i;
            }
        }

        int actualLabel = 0;
        for (int i = 0; i < actualValues.size(); i++) {
            if (actualValues[i] == 1.0) {
                actualLabel = i;
                break;
            }

        }

        if (predictedLabel == actualLabel) {
            totalCorrect++;
        }

        // std::cout << "predicted: [";
        // for (int i = 0; i < actualValues.size(); i++) {
        //     if (i == actualValues.size() - 1) {
        //         std::cout << predictedValues[i] << "]" << std::endl;
        //     } else {
        //         std::cout << predictedValues[i] << ", ";
        //     }
        // }

        // std::cout << "actual: [";
        // for (int i = 0; i < actualValues.size(); i++) {
        //     if (i == actualValues.size() - 1) {
        //         std::cout << actualValues[i] << "]" << std::endl;
        //     } else {
        //         std::cout << actualValues[i] << ", ";
        //     }
        // }

    }

    float accuracy = (1.0 * totalCorrect) / testingData.size();
    std::cout << "Total testing accuracy:" << std::endl;
    std::cout << accuracy << std::endl;

}