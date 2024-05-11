#ifndef COSTFUNCTION_H
#define COSTFUNCTION_H

#include <string>
#include <vector>
#include "Layer.cuh"

class CostFunction {
    private:
        std::string type;
        double getCostBCE();
        
    public:
        CostFunction(std::string type);
        CostFunction(); //default constructor

        double getCost();
        void getErrorFinalLayer(Layer* finalLayer, Matrix& trueLabel, Matrix& gradientCostBias);

        double currentCost; //default initialized to 0
        

};

__global__ void getErrorFinalLayerBCE(float* error, float* a, float* z, float* y, float* g_b, int xDim, int yDim)
#endif
