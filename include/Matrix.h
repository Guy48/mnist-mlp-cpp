#ifndef MATRIX_H
#define MATRIX_H

#include <iosfwd>
#include <cmath>
#include <stdexcept>

struct matrix_dims
{
    int rows;
    int cols;
};

class Matrix
{
private:
    matrix_dims _dims;
    float* _elements;

public:
    explicit Matrix(int rows, int cols);
    Matrix();
    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);
    ~Matrix();

    int get_rows() const;
    int get_cols() const;

    Matrix& transpose();
    Matrix& vectorize();
    void plain_print() const;
    Matrix dot(const Matrix& mat) const;
    float norm() const;
    Matrix rref() const;
    int argmax() const;
    float sum() const;

    Matrix operator+(const Matrix& mat) const;
    Matrix& operator+=(const Matrix& mat);
    Matrix operator*(const Matrix& mat) const;
    friend Matrix operator*(float scalar, const Matrix& mat);
    Matrix operator*(float scalar) const;

    float operator()(int row, int col) const;
    float& operator()(int row, int col);
    float operator[](int index) const;
    float& operator[](int index);

    friend std::ostream& operator<<(std::ostream& s, const Matrix& mat);
    friend std::istream& operator>>(std::istream& s, Matrix& mat);
};

#endif // MATRIX_H
