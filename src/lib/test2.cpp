#include <memory>
class test2 {
    private:

    public:
        test2() {
            hi = std::unique_ptr<float>(new float(3.0));

        }

        std::unique_ptr<float> hi;

};