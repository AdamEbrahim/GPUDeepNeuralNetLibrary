#include "CostFunction.cuh"
#include <iostream>

CostFunction::CostFunction(std::string type) {
    if (type == "BCE") {
        this->type = type;
    } else { //error checking for invalid cost function type
        throw(type);
    }
}

CostFunction::CostFunction() { //default constructor, default to BCE cost function
    this->type = "BCE";
}

void CostFunction::getErrorFinalLayerBCE(Layer* finalLayer, Matrix& trueLabel) {
    float* finalLayerInputError = finalLayer->inputError.valuesDevice.get();
    Layer::act* activationPtr = finalLayer->activation.get();
    Layer::act* activationPrimePtr = finalLayer->activationPrime.get();
    float* a = finalLayer->outputActivation.valuesDevice.get();
    float* z = finalLayer->outputActivationPrime.valuesDevice.get();
    float* y = trueLabel.valuesDevice.get();
}

void CostFunction::getErrorFinalLayer(Layer* finalLayer) {
    if (this->type == "BCE") {
        this->getErrorFinalLayerBCE(finalLayer);
    } else {
        std::cout << "Final layer error can't be computed due to unsupported loss function type" << std::endl;
    }
}

double CostFunction::getCostBCE() {
    double val;

    this->currentCost = val;
    return val;
}

double CostFunction::getCost() {
    if (this->type == "BCE") {
        return this->getCostBCE();
    } else {
        return 0.0;
    }
}