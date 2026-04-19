# MNIST MLP Network

A small C++14 project that loads a trained fully connected MNIST classifier from binary parameter files and predicts digits from binary 28×28 float images.

## Build

```bash
cmake -S . -B build
cmake --build build
```

## Run

```bash
./build/mnist_cli w1 w2 w3 w4 b1 b2 b3 b4
```

The arguments must point to binary `float32` matrices in this order:

- `w1`, `w2`, `w3`, `w4`: weight matrices
- `b1`, `b2`, `b3`, `b4`: bias matrices

After startup, the program asks for an image file path. Type `q` to quit.

## Image format

The image files are expected to contain exactly `28 * 28 = 784` `float32` values in row-major order.

## Notes

The repository is intentionally kept as a small inference demo. The original course-specific build files and submission helpers are not included in this version.
