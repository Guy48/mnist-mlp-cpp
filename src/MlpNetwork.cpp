#include "MlpNetwork.h"

MlpNetwork::MlpNetwork(Matrix* weights, Matrix* biases)
    : _l1(weights[0], biases[0], activation::relu),
      _l2(weights[1], biases[1], activation::relu),
      _l3(weights[2], biases[2], activation::relu),
      _l4(weights[3], biases[3], activation::softmax)
{
}

digit MlpNetwork::operator()(const Matrix& img) const
{
    Matrix input(img);
    input.vectorize();
    Matrix output = _l4(_l3(_l2(_l1(input))));
    const unsigned int value = static_cast<unsigned int>(output.argmax());
    return digit{value, output[value]};
}
