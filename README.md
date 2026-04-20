# MNIST MLP Classifier

A lightweight C++ implementation of a fully connected neural network for classifying handwritten digits from the MNIST dataset.

The project loads a set of pre-trained parameters from binary files, processes a single 28×28 grayscale image, and returns the predicted digit together with the model confidence.

## Features

- Custom matrix implementation with dynamic storage
- Dense layers with ReLU and Softmax activations
- Binary loading of pretrained weights, biases, and input images
- Command-line interface for interactive inference
- Optional Python helper for visualizing raw image files

## Project Structure

```text
include/   Public headers for the matrix, activation, dense, and network components
src/       C++ source files and the CLI entry point
tools/     Utility scripts, including image visualization
```

## Requirements

- A C++14-compatible compiler
- CMake 3.22 or newer
- Python 3 with `numpy` and `matplotlib` for the optional visualization script

## Build

From the project root:

```bash
cmake -S . -B build
cmake --build build
```

This produces the main executable in the build directory.

## Run

The program expects eight binary parameter files as command-line arguments:

- four weight matrices: `w1`, `w2`, `w3`, `w4`
- four bias vectors: `b1`, `b2`, `b3`, `b4`

Example:

```bash
./build/mnist_cli parameters/w1.bin parameters/w2.bin parameters/w3.bin parameters/w4.bin \
                 parameters/b1.bin parameters/b2.bin parameters/b3.bin parameters/b4.bin
```

After startup, the application prompts for an image path. Enter the path to a binary MNIST image file stored as `float32` values in row-major order.

To exit the interactive loop, type:

```text
q
```

## Image Format

The input image files are expected to contain exactly:

- `28 × 28 = 784` values
- data type: `float32`
- layout: row-major
- raw binary encoding, without headers

The same format is used by the parameter files as well.

## Previewing an Image

To visualize a raw image file, use the helper script:

```bash
python tools/plot_img.py images/<image_file>
```

## Notes

- The network architecture is fixed and defined in `include/MlpNetwork.h`.
- The parameter and image files are not human-readable text files; they are binary blobs written directly from floating-point arrays.