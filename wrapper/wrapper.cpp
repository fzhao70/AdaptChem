#include <torch/script.h>
#include <iostream>
#include <memory>
#include <vector>
#include <cstring>

// Function declarations and implementations
extern "C" {

    // Load the TorchScript model
    torch::jit::script::Module* load_model(const char* model_file) {
        try {
            return new torch::jit::script::Module(torch::jit::load(model_file));
        } catch (const c10::Error& e) {
            std::cerr << "Error loading the model: " << e.what() << std::endl;
            return nullptr;
        }
    }

    // Destroy the model to free memory
    void destroy_model(torch::jit::script::Module* model) {
        delete model;
    }

    // Forward function
    void forward(torch::jit::script::Module* model,
                 float* input_data, int64_t* input_dims, int64_t input_dims_size,
                 float* output_data, int64_t* output_dims, int64_t output_dims_size) {

        // Create input tensor
        std::vector<int64_t> input_sizes(input_dims, input_dims + input_dims_size);
        torch::Tensor input_tensor = torch::from_blob(input_data, input_sizes).clone();

        // Run forward pass
        at::Tensor output = model->forward({input_tensor}).toTensor();

        // Copy output data
        std::memcpy(output_data, output.data_ptr(), output.numel() * sizeof(float));

        // Copy output dimensions
        auto output_sizes = output.sizes();
        int64_t min_size = std::min(output_dims_size, static_cast<int64_t>(output_sizes.size()));
        for (int64_t i = 0; i < min_size; ++i) {
            output_dims[i] = output_sizes[i];
        }
    }
}