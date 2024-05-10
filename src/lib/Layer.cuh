#ifndef LAYER_H
#define LAYER_H

#include "Matrix.cuh"

class Layer { //interface for a NN layer
    private:

    protected:
        Layer(int prevNumNeurons, int numNeurons);
        virtual void callGetActivation(dim3 blocks, dim3 threads, float* w, float* x, float* a, float* b, float* z, int xDim, int yDim) = 0;
        virtual void callBackPropError(dim3 blocks, dim3 threads, float* nextError, float* w, float* z, float* error, int xDim, int yDim) = 0;

    public:
        void forwardPass(Matrix& prevLayerActivations);
        void backprop(Matrix& nextLayerError, Matrix& nextLayerWeights);
        void initializeMatrices();
        virtual ~Layer() = 0; //Destructor

        Matrix weights;
        Matrix biases;
        Matrix outputActivationPrime;
        Matrix outputActivation;
        Matrix inputError;

        int numberNeurons;
        int prevLayerNumNeurons;

        typedef float (*act)(float);

        std::shared_ptr<act> activation; //function pointer to device function activation; set in device
        std::shared_ptr<act> activationPrime;  //function pointer to device function activationPrime; set in device

};

inline Layer::~Layer() {} //Inline definition of destructor, inline means sub the body of this function whenever
                          //the function is called. This means will just sub nothing whenever call Layer destructor

#endif