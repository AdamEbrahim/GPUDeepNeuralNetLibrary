#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include "lib/GPUNeuralNetwork.cuh"

int reverseInt (int i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

std::vector<std::unique_ptr<std::vector<float> > >* read_mnist_image_data(std::string path)
{
    std::vector<std::unique_ptr<std::vector<float> > >* allTrainingData;

    //read Training images file -- get training data
    std::ifstream file (path, std::ios::binary);
    if (file.is_open())
    {
        std::cout << "hi" << std::endl;
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);

        std::cout << number_of_images << std::endl;
        std::cout << n_rows << std::endl;
        std::cout << n_cols << std::endl;

        allTrainingData = new std::vector<std::unique_ptr<std::vector<float> > > (number_of_images);

        for(int i=0;i<number_of_images;++i)
        {
            std::vector<float>* image_data = new std::vector<float> (n_rows * n_cols, 0); //initialize to n_rows * n_cols of 0
            (*allTrainingData)[i] = std::unique_ptr<std::vector<float>> (image_data);

            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    (*image_data)[(r * n_cols) + c] = temp / 255.0; //normalize pixel data to range [0,1]

                }
            }
        }
        
    } else {
        std::cout << "unable to read file" << std::endl;
    }

    return allTrainingData;
}

std::vector<std::unique_ptr<std::vector<float> > >* read_mnist_label_data(std::string path)
{
    std::vector<std::unique_ptr<std::vector<float> > >* trueLabels;

    //read Label images file -- get label data
    std::ifstream file (path, std::ios::binary);
    if (file.is_open())
    {
        std::cout << "hi" << std::endl;
        int magic_number=0;
        int number_of_images=0;
        file.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);

        std::cout << number_of_images << std::endl;

        trueLabels = new std::vector<std::unique_ptr<std::vector<float> > > (number_of_images);

        for(int i=0;i<number_of_images;++i)
        {
            std::vector<float>* image_data = new std::vector<float> (10, 0); //initialize to 10 (10 different labels) of 0
            (*trueLabels)[i] = std::unique_ptr<std::vector<float>> (image_data);

            unsigned char temp=0;
            file.read((char*)&temp,sizeof(temp));
            (*image_data)[temp] = 1; //set the true label in the vector to be 1

        }
        
    } else {
        std::cout << "unable to read file" << std::endl;
    }

    return trueLabels;
}

int main() {
    std::vector<std::unique_ptr<std::vector<float> > >* trainingImageData = read_mnist_image_data("src/data/train-images-idx3-ubyte");
    std::vector<std::unique_ptr<std::vector<float> > >* trainingLabelData = read_mnist_label_data("src/data/train-labels-idx1-ubyte");
    std::vector<std::unique_ptr<std::vector<float> > >* testingImageData = read_mnist_image_data("src/data/t10k-images-idx3-ubyte");
    std::vector<std::unique_ptr<std::vector<float> > >* testingLabelData = read_mnist_label_data("src/data/t10k-labels-idx1-ubyte");

    std::vector<std::string> layerTypes;
    layerTypes.emplace_back("Sigmoid");
    layerTypes.emplace_back("Sigmoid");
    layerTypes.emplace_back("Sigmoid"); //final layer = sigmoid, coupled with binary cross entropy loss function
    std::vector<int> layerCounts;
    layerCounts.emplace_back(50);
    layerCounts.emplace_back(30);
    layerCounts.emplace_back(10);

    GPUNeuralNetwork test("BCE", (*((*trainingImageData)[0])).size(), (*((*trainingLabelData)[0])).size(), 3.0);
    test.initializeLayers(layerTypes, layerCounts);
    test.trainNetwork(4, *trainingImageData, *trainingLabelData, 10);
    test.testNetwork(*testingImageData, *testingLabelData);

    delete trainingImageData;
    delete trainingLabelData;
    delete testingImageData;
    delete testingLabelData;
    return 0;
}