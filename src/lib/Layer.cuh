#ifndef LAYER_H
#define LAYER_H

#include "Matrix.cuh"

class Layer { //interface for a NN layer
    private:

    protected:
        Layer(int prevNumNeurons, int numNeurons);
        virtual void forwardPass(Matrix& prevLayerActivations) = 0;
        virtual void backprop(Matrix& nextLayerError, Matrix& nextLayerWeights, Matrix& prevLayerActivations) = 0;

    public:
        void initializeMatrices();
        virtual ~Layer() = 0; //Destructor

        Matrix weights;
        Matrix biases;
        Matrix outputActivation;
        Matrix inputError;

        int numberNeurons;
        int prevLayerNumNeurons;

};

inline Layer::~Layer() {} //Inline definition of destructor, inline means sub the body of this function whenever
                          //the function is called. This means will just sub nothing whenever call Layer destructor

#endif