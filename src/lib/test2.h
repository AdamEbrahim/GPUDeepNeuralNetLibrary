#ifndef TEST2_H
#define TEST2_H

#include <memory>

float other(float m);

class test2 {
    private:

    public:

        test2();

        //typedef float (*activationPrime)(float);
        typedef float (*activation)(float);
        std::shared_ptr<activation> act;
        //std::shared_ptr<activationPrime> actPrime;

        std::unique_ptr<float> hi;

};
#endif