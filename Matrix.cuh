#include <memory>

class Matrix {
    private:

    public:
        Matrix(int xDim, int yDim); //flattened array of values, xDim * yDim = size of values
        int xDim, yDim;
        std::shared_ptr<float[]> valuesHost;
        std::shared_ptr<float> valuesDevice;

        void allocateCUDAMemory();
        void allocateHostMemory();

        
};