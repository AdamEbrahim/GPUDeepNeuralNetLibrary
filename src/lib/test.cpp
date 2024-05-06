#include <memory>
#include <vector>
#include <iostream>
#include <random>

// void test_unique_ptr_should_fail(std::vector<std::unique_ptr<int> > hi) {
//     for (int i = 0; i < hi.size(); i++) {
//         std::cout << *(hi[i]) << std::endl;
//     }
// }

// void test_unique_ptr(std::vector<std::unique_ptr<int> >& hi) {
//     for (int i = 0; i < hi.size(); i++) {
//         std::cout << *(hi[i]) << std::endl;
//     }

// }

// void test_add(std::vector<std::unique_ptr<int> >& hi) {
//     hi.push_back(std::unique_ptr<int>(new int(5)));
//     hi.push_back(std::unique_ptr<int>(new int(7)));
//     hi.push_back(std::unique_ptr<int>(new int(9)));
// }

//Mini batch will call runTrainingExample() on all training inputs in mini batch of size m, use that to perform gradient descent
//inputData is a vector of size m, where each element inputData[m] is a vector of one training example's input layer encodings 
void runMiniBatch(std::vector<std::unique_ptr<std::vector<float> > >& inputData) {
    int miniBatchSize = inputData.size();
    for (int i = 0; i < inputData.size(); i++) {
        std::cout << "[" << (*(inputData[i]))[0] << ", " << (*(inputData[i]))[1] << ", " << (*(inputData[i]))[2] << "]" << std::endl;
    }
}

void randomizeMiniBatches(std::vector<std::unique_ptr<std::vector<float> > >& allTrainingData, std::vector<std::vector<std::unique_ptr<std::vector<float> > > >& miniBatches, int miniBatchSize, std::default_random_engine& rng) {
    std::shuffle(allTrainingData.begin(), allTrainingData.end(), rng);

    for (int i = 0; i < allTrainingData.size(); i++) {
        int currBatchIdx = i / miniBatchSize;
        miniBatches[currBatchIdx].push_back(std::move(allTrainingData[i]));
    }
    //clear all the remaining nullptrs after move
    allTrainingData.clear();
}

void trainNetwork(int numEpochs, std::vector<std::unique_ptr<std::vector<float> > >& allTrainingData, int miniBatchSize) {
    int numMiniBatches = (allTrainingData.size() / miniBatchSize) + 1;
    std::vector<std::vector<std::unique_ptr<std::vector<float> > > > miniBatches;
    for (int i = 0; i < numMiniBatches; i++) {
        miniBatches.push_back(std::vector<std::unique_ptr<std::vector<float> > >(0));
    }
    auto rd = std::random_device {}; //randomly seed the random generator used for vector shuffle
    auto rng = std::default_random_engine {rd()}; //create a reusable instance of default random engine, rd() is function call operator overloading

    for (int i = 0; i < numEpochs; i++) {
        std::cout << "Beginning Epoch " << i << " of training:" << std::endl;

        //Move everything from mini batch vectors back to allTrainingData vector
        if (i != 0) {
            for (int j = 0; j < numMiniBatches; j++) {
                for (int k = 0; k < miniBatches[j].size(); k++) {
                    allTrainingData.push_back(std::move(miniBatches[j][k]));
                }
                //clear all the remaining nullptrs after move
                miniBatches[j].clear();
            }

        }

        randomizeMiniBatches(allTrainingData, miniBatches, miniBatchSize, rng);
        for (int j = 0; j < numMiniBatches; j++) {
            std::cout << "Running Mini Batch " << j << std::endl;
            runMiniBatch(miniBatches[j]);
        }
        
    }

}

int main() {
    // std::vector<std::unique_ptr<int> > tester;
    // test_add(tester);

    // test_unique_ptr(tester);

    // std::vector<std::unique_ptr<int> > tester2;
    //tester2.push_back(tester[0]); THROWS ERROR BECAUSE CANT COPY UNIQUE_PTR
    // tester2.push_back(std::move(tester[0]));
    // std::cout << *(tester2[0]) << std::endl;
    //std::cout << *(tester[0]) << std::endl; SEGFAULT BECAUSE TESTER[0] HAS BEEN MOVED TO TESTER2[0]. TESTER[0] IS NOW NULLPTR

    std::vector<std::unique_ptr<std::vector<float> > > allTrainingData;
    allTrainingData.push_back(std::unique_ptr<std::vector<float> > (new std::vector<float>{1, 2, 3}));
    allTrainingData.push_back(std::unique_ptr<std::vector<float> > (new std::vector<float>{2, 3, 4}));
    allTrainingData.push_back(std::unique_ptr<std::vector<float> > (new std::vector<float>{3, 4, 5}));
    allTrainingData.push_back(std::unique_ptr<std::vector<float> > (new std::vector<float>{4, 5, 6}));
    allTrainingData.push_back(std::unique_ptr<std::vector<float> > (new std::vector<float>{5, 6, 7}));
    allTrainingData.push_back(std::unique_ptr<std::vector<float> > (new std::vector<float>{6, 7, 8}));
    allTrainingData.push_back(std::unique_ptr<std::vector<float> > (new std::vector<float>{7, 8, 9}));
    allTrainingData.push_back(std::unique_ptr<std::vector<float> > (new std::vector<float>{8, 9, 10}));
    allTrainingData.push_back(std::unique_ptr<std::vector<float> > (new std::vector<float>{9, 10, 11}));
    allTrainingData.push_back(std::unique_ptr<std::vector<float> > (new std::vector<float>{10, 11, 12}));
    allTrainingData.push_back(std::unique_ptr<std::vector<float> > (new std::vector<float>{11, 12, 13}));

    trainNetwork(3, allTrainingData, 4);

    return 0;
}