#include "test2.h"
#include <memory>

float other(float m) {
    return 5.0;
}

test2::test2() {
    hi = std::unique_ptr<float>(new float(3.0));
    activation* o = (activation*) malloc(1 * sizeof(activation));
    act = std::shared_ptr<activation>(o);
    *act = other;
}
