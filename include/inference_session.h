#pragma once

#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace nb = nanobind;

class InferenceSession {
public:
	enum class InferenceSessionProvider {
		CPU,
		GPU
	};

	struct SessionInputOutput {
		std::string name;
		size_t rank;
		std::vector<int64_t> shape;
		ONNXTensorElementDataType type;
	};

	InferenceSession(const char *model_path,
		InferenceSessionProvider provider,
		int intra_op_num_threads);

	std::vector<SessionInputOutput> get_inputs();
	std::vector<SessionInputOutput> get_outputs();

	std::vector<nb::ndarray<nb::numpy>> run(const std::vector<nb::ndarray<nb::numpy>> &inputs);

private:
	Ort::Session create_session(const char *model_path,
		InferenceSessionProvider provider,
		int intra_op_num_threads);
	void validate_input(const std::vector<nb::ndarray<nb::numpy>> &inputs);

	std::unordered_map<ONNXTensorElementDataType, nb::dlpack::dtype> m_dtype_conversion_map = {
		{ ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, nb::dtype<float>() },
		{ ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, nb::dtype<uint16_t>() },
		{ ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, nb::dtype<double>() },
		{ ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, nb::dtype<int32_t>() },
		{ ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, nb::dtype<int64_t>() },
		{ ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, nb::dtype<int16_t>() },
		{ ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, nb::dtype<int8_t>() },
		{ ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, nb::dtype<uint8_t>() },
		{ ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, nb::dtype<bool>() },
	};

	std::vector<SessionInputOutput> m_inputs;
	std::vector<SessionInputOutput> m_outputs;
	Ort::Env m_ort_env;
	Ort::Session m_session;
	Ort::MemoryInfo m_memory_info;
};
