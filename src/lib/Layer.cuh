#include "Matrix.cuh"

class Layer { //interface for a NN layer
    private:

    public:
        Layer(int prevNumNeurons, int numNeurons) : weights{prevNumNeurons, numNeurons}, biases{1, numNeurons}, outputActivation{1, numNeurons}, inputError{1, numNeurons} {

        }
        virtual ~Layer() = 0; //Destructor
        virtual void forwardPass(Matrix& prevLayerActivations) = 0;
        virtual void backprop(Matrix& nextLayerError, Matrix& nextLayerWeights, Matrix& prevLayerActivations) = 0;

        Matrix weights;
        Matrix biases;
        Matrix outputActivation;
        Matrix inputError;

        int numberNeurons;

};

inline Layer::~Layer() {} //Inline definition of destructor, inline means sub the body of this function whenever
                          //the function is called. This means will just sub nothing whenever call Layer destructor