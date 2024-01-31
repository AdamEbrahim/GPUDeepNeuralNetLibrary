#include "CostFunction.cuh"

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