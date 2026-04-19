#include "Matrix.h"

#include <algorithm>
#include <iostream>
#include <utility>

namespace
{
constexpr float kAsciiThreshold = 0.1f;
constexpr char kOutOfRange[] = "Out of range";
constexpr char kDimMismatch[] = "Matrix dimensions are not compatible";
}

Matrix::Matrix(int rows, int cols)
    : _dims{rows, cols}, _elements(new float[rows * cols])
{
    for (int i = 0; i < rows * cols; ++i)
    {
        _elements[i] = 0.0f;
    }
}

Matrix::Matrix() : Matrix(1, 1)
{
}

Matrix::Matrix(const Matrix& other)
    : Matrix(other._dims.rows, other._dims.cols)
{
    std::copy(other._elements,
              other._elements + (_dims.rows * _dims.cols),
              _elements);
}

Matrix& Matrix::operator=(const Matrix& other)
{
    if (this == &other)
    {
        return *this;
    }

    Matrix tmp(other);
    std::swap(_dims, tmp._dims);
    std::swap(_elements, tmp._elements);
    return *this;
}

Matrix::~Matrix()
{
    delete[] _elements;
}

int Matrix::get_rows() const
{
    return _dims.rows;
}

int Matrix::get_cols() const
{
    return _dims.cols;
}

Matrix& Matrix::transpose()
{
    Matrix transposed(_dims.cols, _dims.rows);
    for (int r = 0; r < _dims.rows; ++r)
    {
        for (int c = 0; c < _dims.cols; ++c)
        {
            transposed(c, r) = (*this)(r, c);
        }
    }
    *this = transposed;
    return *this;
}

Matrix& Matrix::vectorize()
{
    _dims.rows *= _dims.cols;
    _dims.cols = 1;
    return *this;
}

void Matrix::plain_print() const
{
    for (int r = 0; r < _dims.rows; ++r)
    {
        for (int c = 0; c < _dims.cols; ++c)
        {
            std::cout << (*this)(r, c) << ' ';
        }
        std::cout << '\n';
    }
}

Matrix Matrix::dot(const Matrix& mat) const
{
    if (_dims.rows != mat._dims.rows || _dims.cols != mat._dims.cols)
    {
        throw std::length_error(kDimMismatch);
    }

    Matrix result(*this);
    for (int i = 0; i < _dims.rows * _dims.cols; ++i)
    {
        result._elements[i] *= mat._elements[i];
    }
    return result;
}

float Matrix::norm() const
{
    float sum_squares = 0.0f;
    for (int i = 0; i < _dims.rows * _dims.cols; ++i)
    {
        sum_squares += _elements[i] * _elements[i];
    }
    return std::sqrt(sum_squares);
}

int Matrix::argmax() const
{
    int index = 0;
    float max_value = _elements[0];
    for (int i = 1; i < _dims.rows * _dims.cols; ++i)
    {
        if (_elements[i] > max_value)
        {
            max_value = _elements[i];
            index = i;
        }
    }
    return index;
}

float Matrix::sum() const
{
    float total = 0.0f;
    for (int i = 0; i < _dims.rows * _dims.cols; ++i)
    {
        total += _elements[i];
    }
    return total;
}

Matrix Matrix::operator+(const Matrix& mat) const
{
    if (_dims.rows != mat._dims.rows || _dims.cols != mat._dims.cols)
    {
        throw std::length_error(kDimMismatch);
    }

    Matrix result(*this);
    for (int i = 0; i < _dims.rows * _dims.cols; ++i)
    {
        result._elements[i] += mat._elements[i];
    }
    return result;
}

Matrix& Matrix::operator+=(const Matrix& mat)
{
    if (_dims.rows != mat._dims.rows || _dims.cols != mat._dims.cols)
    {
        throw std::length_error(kDimMismatch);
    }

    for (int i = 0; i < _dims.rows * _dims.cols; ++i)
    {
        _elements[i] += mat._elements[i];
    }
    return *this;
}

Matrix Matrix::operator*(const Matrix& mat) const
{
    if (_dims.cols != mat._dims.rows)
    {
        throw std::length_error(kDimMismatch);
    }

    Matrix result(_dims.rows, mat._dims.cols);
    for (int r = 0; r < _dims.rows; ++r)
    {
        for (int c = 0; c < mat._dims.cols; ++c)
        {
            float cell = 0.0f;
            for (int k = 0; k < _dims.cols; ++k)
            {
                cell += (*this)(r, k) * mat(k, c);
            }
            result(r, c) = cell;
        }
    }
    return result;
}

Matrix operator*(float scalar, const Matrix& mat)
{
    Matrix result(mat);
    for (int i = 0; i < result.get_rows() * result.get_cols(); ++i)
    {
        result[i] *= scalar;
    }
    return result;
}

Matrix Matrix::operator*(float scalar) const
{
    return scalar * (*this);
}

float Matrix::operator()(int row, int col) const
{
    if (row < 0 || col < 0 || row >= _dims.rows || col >= _dims.cols)
    {
        throw std::out_of_range(kOutOfRange);
    }
    return _elements[row * _dims.cols + col];
}

float& Matrix::operator()(int row, int col)
{
    if (row < 0 || col < 0 || row >= _dims.rows || col >= _dims.cols)
    {
        throw std::out_of_range(kOutOfRange);
    }
    return _elements[row * _dims.cols + col];
}

float Matrix::operator[](int index) const
{
    if (index < 0 || index >= _dims.rows * _dims.cols)
    {
        throw std::out_of_range(kOutOfRange);
    }
    return _elements[index];
}

float& Matrix::operator[](int index)
{
    if (index < 0 || index >= _dims.rows * _dims.cols)
    {
        throw std::out_of_range(kOutOfRange);
    }
    return _elements[index];
}

std::ostream& operator<<(std::ostream& s, const Matrix& mat)
{
    for (int r = 0; r < mat._dims.rows; ++r)
    {
        for (int c = 0; c < mat._dims.cols; ++c)
        {
            if (mat(r, c) > kAsciiThreshold)
            {
                s << "**";
            }
            else
            {
                s << "  ";
            }
        }
        s << '\n';
    }
    return s;
}

std::istream& operator>>(std::istream& s, Matrix& mat)
{
    s.read(reinterpret_cast<char*>(mat._elements),
           static_cast<std::streamsize>(mat._dims.rows * mat._dims.cols * sizeof(float)));
    if (!s)
    {
        throw std::runtime_error(kOutOfRange);
    }
    return s;
}

// Bonus utility kept for completeness.
Matrix Matrix::rref() const
{
    Matrix res(*this);
    int lead = 0;
    for (int r = 0; r < res._dims.rows; ++r)
    {
        if (lead >= res._dims.cols)
        {
            break;
        }

        int pivot = r;
        while (res(pivot, lead) == 0.0f)
        {
            ++pivot;
            if (pivot >= res._dims.rows)
            {
                pivot = r;
                ++lead;
                if (lead >= res._dims.cols)
                {
                    return res;
                }
            }
        }

        if (pivot != r)
        {
            for (int c = 0; c < res._dims.cols; ++c)
            {
                std::swap(res(pivot, c), res(r, c));
            }
        }

        const float leading = res(r, lead);
        if (leading != 0.0f)
        {
            for (int c = 0; c < res._dims.cols; ++c)
            {
                res(r, c) /= leading;
            }
        }

        for (int i = 0; i < res._dims.rows; ++i)
        {
            if (i != r)
            {
                const float scale = res(i, lead);
                if (scale != 0.0f)
                {
                    for (int c = 0; c < res._dims.cols; ++c)
                    {
                        res(i, c) -= scale * res(r, c);
                    }
                }
            }
        }
        ++lead;
    }
    return res;
}
