#ifndef MLPNETWORK_H
#define MLPNETWORK_H

#include "Dense.h"

constexpr int MLP_SIZE = 4;

struct digit
{
    unsigned int value;
    float probability;
};

static const matrix_dims img_dims{28, 28};
static const matrix_dims weights_dims[MLP_SIZE] = {
    {128, 784},
    {64, 128},
    {20, 64},
    {10, 20}
};
static const matrix_dims bias_dims[MLP_SIZE] = {
    {128, 1},
    {64, 1},
    {20, 1},
    {10, 1}
};

class MlpNetwork
{
private:
    Dense _l1;
    Dense _l2;
    Dense _l3;
    Dense _l4;

public:
    MlpNetwork(Matrix* weights, Matrix* biases);
    digit operator()(const Matrix& img) const;
};

#endif // MLPNETWORK_H
