#ifndef DENSE_H
#define DENSE_H

#include "Activation.h"

typedef Matrix (*act_func)(const Matrix& mat);

class Dense
{
private:
    Matrix _weights;
    Matrix _bias;
    act_func _activationFunction;

public:
    Dense(const Matrix& weights, const Matrix& bias, act_func a_f);
    const Matrix& get_weights() const;
    const Matrix& get_bias() const;
    act_func get_activation() const;
    Matrix operator()(const Matrix& mat) const;
};

#endif // DENSE_H
