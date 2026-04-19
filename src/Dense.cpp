#include "Dense.h"

Dense::Dense(const Matrix& weights, const Matrix& bias, act_func a_f)
    : _weights(weights), _bias(bias), _activationFunction(a_f)
{
}

const Matrix& Dense::get_weights() const
{
    return _weights;
}

const Matrix& Dense::get_bias() const
{
    return _bias;
}

act_func Dense::get_activation() const
{
    return _activationFunction;
}

Matrix Dense::operator()(const Matrix& mat) const
{
    Matrix out = _weights * mat;
    out += _bias;
    return _activationFunction(out);
}
