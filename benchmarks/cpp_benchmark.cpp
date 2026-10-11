// Direct-C++ benchmark for the cpp-onnx-python core.
//
// Emits a single JSON object with timing statistics so that the Python
// benchmark harness (or a human) can compare the three paths on one workload.
//
// Build:
//   cmake -S . -B build-native
//   cmake --build build-native --target cpp_onnx_python_benchmark
// Run:
//   ./build-native/cpp_onnx_python_benchmark tests/fixtures/mul_add.onnx 50 10

#include "cpp_onnx_python/inference.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace cpp_onnx_python;

int main(int argc, char **argv)
{
	const std::string model_path =
		argc > 1 ? argv[1] : "tests/fixtures/mul_add.onnx";
	const int repetitions = argc > 2 ? std::stoi(argv[2]) : 50;
	const int warmup = argc > 3 ? std::stoi(argv[3]) : 10;

	try {
		InferenceSession session(model_path, Provider::CPU);
		const TensorInfo &info = session.inputs().at(0);

		std::size_t element_count = 1;
		for (int64_t dim : info.shape)
			element_count *= static_cast<std::size_t>(dim > 0 ? dim : 1);

		std::vector<float> data(element_count);
		for (std::size_t i = 0; i < element_count; ++i)
			data[i] = std::sin(static_cast<float>(i) * 0.37f) * 2.0f;

		InputTensor input;
		input.data = data.data();
		input.type = DataType::Float;
		input.shape = info.shape.data();
		input.rank = info.shape.size();
		input.nbytes = data.size() * sizeof(float);

		const std::vector<InputTensor> inputs { input };

		for (int i = 0; i < warmup; ++i)
			(void)session.run(inputs);

		std::vector<double> samples;
		samples.reserve(static_cast<std::size_t>(repetitions));
		for (int i = 0; i < repetitions; ++i) {
			const auto start = std::chrono::steady_clock::now();
			(void)session.run(inputs);
			const auto finish = std::chrono::steady_clock::now();
			samples.push_back(
				std::chrono::duration<double, std::milli>(finish - start).count());
		}

		std::sort(samples.begin(), samples.end());
		const double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
		const double mean = sum / static_cast<double>(samples.size());
		const double median = samples[samples.size() / 2];
		const double p95 =
			samples[static_cast<std::size_t>(0.95 * (samples.size() - 1))];

		std::cout << "{\n";
		std::cout << "  \"path\": \"direct_cpp\",\n";
		std::cout << "  \"model\": \"" << model_path << "\",\n";
		std::cout << "  \"provider\": \"cpu\",\n";
		std::cout << "  \"input_elements\": " << element_count << ",\n";
		std::cout << "  \"warmup\": " << warmup << ",\n";
		std::cout << "  \"repetitions\": " << repetitions << ",\n";
		std::cout << "  \"mean_ms\": " << mean << ",\n";
		std::cout << "  \"median_ms\": " << median << ",\n";
		std::cout << "  \"p95_ms\": " << p95 << ",\n";
		std::cout << "  \"throughput_per_s\": " << 1000.0 / mean << "\n";
		std::cout << "}\n";
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "error: " << error.what() << "\n";
		return 1;
	}
}
