#include <string>
#include <vector>

class CostFunction {
    private:
        std::string type;
        double getCostBCE();
        double getErrorFinalLayerBCE();
        
    public:
        CostFunction(std::string type);
        CostFunction(); //default constructor

        double getCost();
        double getErrorFinalLayer();

        double currentCost;
        

};