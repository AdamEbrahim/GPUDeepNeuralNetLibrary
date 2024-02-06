#include "Matrix.cuh"
#include "Layer.cuh"

class SigmoidLayer: public Layer {
    private:
    __global__ void getActivation(float* x);

    public:
        SigmoidLayer(int prevNumNeurons, int numNeurons);
        void forwardPass(Matrix& prevLayerActivations);
        void backprop(Matrix& nextLayerError, Matrix& nextLayerWeights, Matrix& prevLayerActivations);

};