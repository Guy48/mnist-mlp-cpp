#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "Matrix.h"

namespace activation {
    Matrix relu(const Matrix& mat);
    Matrix softmax(const Matrix& mat);
}

#endif // ACTIVATION_H
