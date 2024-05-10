#ifndef COSTFUNCTION_H
#define COSTFUNCTION_H

#include <string>
#include <vector>
#include "Layer.cuh"

class CostFunction {
    private:
        std::string type;
        double getCostBCE();
        void getErrorFinalLayerBCE(Layer* finalLayer);
        
    public:
        CostFunction(std::string type);
        CostFunction(); //default constructor

        double getCost();
        void getErrorFinalLayer(Layer* finalLayer);

        double currentCost; //default initialized to 0
        

};
#endif
