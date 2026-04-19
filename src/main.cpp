#include "MlpNetwork.h"

#include <fstream>
#include <iostream>
#include <string>

namespace
{
constexpr int kExpectedArgc = 1 + (MLP_SIZE * 2);
constexpr const char* kQuitCommand = "q";

void print_usage(const char* program_name)
{
    std::cout << "Usage:\n"
              << "  " << program_name << " w1 w2 w3 w4 b1 b2 b3 b4\n"
              << "\n"
              << "Each argument must be a binary file containing a float32 matrix.\n"
              << "After startup, enter an image file path to classify it, or 'q' to quit.\n";
}

bool readFileToMatrix(const std::string& file_path, Matrix& mat)
{
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open())
    {
        return false;
    }

    try
    {
        file >> mat;
    }
    catch (const std::exception&)
    {
        return false;
    }
    return true;
}

bool loadParameters(int argc, char** argv, Matrix weights[MLP_SIZE], Matrix biases[MLP_SIZE])
{
    for (int i = 0; i < MLP_SIZE; ++i)
    {
        weights[i] = Matrix(weights_dims[i].rows, weights_dims[i].cols);
        biases[i] = Matrix(bias_dims[i].rows, bias_dims[i].cols);

        const std::string weights_path = argv[1 + i];
        const std::string bias_path = argv[1 + MLP_SIZE + i];

        if (!readFileToMatrix(weights_path, weights[i]))
        {
            std::cerr << "Error: could not read weights file for layer " << (i + 1)
                      << ": " << weights_path << '\n';
            return false;
        }
        if (!readFileToMatrix(bias_path, biases[i]))
        {
            std::cerr << "Error: could not read bias file for layer " << (i + 1)
                      << ": " << bias_path << '\n';
            return false;
        }
    }
    return true;
}

void run_cli(MlpNetwork& mlp)
{
    Matrix img(img_dims.rows, img_dims.cols);
    std::string img_path;

    while (true)
    {
        std::cout << "Please insert image path:" << std::endl;
        if (!std::getline(std::cin, img_path))
        {
            break;
        }

        if (img_path == kQuitCommand)
        {
            break;
        }

        if (!readFileToMatrix(img_path, img))
        {
            std::cerr << "Error: invalid image path or size: " << img_path << '\n';
            continue;
        }

        Matrix img_vec = img;
        const digit output = mlp(img_vec.vectorize());

        std::cout << "Image processed:" << std::endl
                  << img << std::endl
                  << "Mlp result: " << output.value
                  << " at probability: " << output.probability << std::endl;
    }
}
}

int main(int argc, char** argv)
{
    if (argc != kExpectedArgc)
    {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    Matrix weights[MLP_SIZE];
    Matrix biases[MLP_SIZE];

    if (!loadParameters(argc, argv, weights, biases))
    {
        return EXIT_FAILURE;
    }

    MlpNetwork mlp(weights, biases);
    run_cli(mlp);
    return EXIT_SUCCESS;
}
