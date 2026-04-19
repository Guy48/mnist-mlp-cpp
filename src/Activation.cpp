#include "Activation.h"

#include <algorithm>
#include <cmath>

namespace activation {

Matrix relu(const Matrix& mat)
{
    Matrix out(mat);
    const int size = out.get_rows() * out.get_cols();
    for (int i = 0; i < size; ++i)
    {
        if (out[i] < 0.0f)
        {
            out[i] = 0.0f;
        }
    }
    return out;
}

Matrix softmax(const Matrix& mat)
{
    Matrix out(mat);
    const int size = out.get_rows() * out.get_cols();

    float max_val = out[0];
    for (int i = 1; i < size; ++i)
    {
        max_val = std::max(max_val, out[i]);
    }

    float denom = 0.0f;
    for (int i = 0; i < size; ++i)
    {
        out[i] = std::exp(out[i] - max_val);
        denom += out[i];
    }

    for (int i = 0; i < size; ++i)
    {
        out[i] /= denom;
    }

    return out;
}

} // namespace activation
