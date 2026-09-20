#include "inference_session.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include <cstdint>
#include <cstring>
#include <format>
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include <stdexcept>

InferenceSession::InferenceSession(const char *model_path,
	InferenceSessionProvider provider,
	int intra_op_num_threads)
	: m_ort_env(ORT_LOGGING_LEVEL_WARNING, "inference_session")
	, m_session(create_session(model_path, provider, intra_op_num_threads))
	, m_memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
	Ort::AllocatorWithDefaultOptions allocator;
	size_t input_count = m_session.GetInputCount();
	size_t output_count = m_session.GetOutputCount();
	for (size_t input_index = 0; input_index < input_count; input_index++) {
		Ort::AllocatedStringPtr name = m_session.GetInputNameAllocated(input_index, allocator);
		Ort::TypeInfo type_info = m_session.GetInputTypeInfo(input_index);
		Ort::ConstTensorTypeAndShapeInfo tensor_info = type_info.GetTensorTypeAndShapeInfo();
		ONNXTensorElementDataType element_type = tensor_info.GetElementType();
		std::vector<int64_t> shape = tensor_info.GetShape();
		m_inputs.emplace_back(std::string(name.get()), shape.size(), shape, element_type);
	}
	for (size_t output_index = 0; output_index < output_count; output_index++) {
		Ort::AllocatedStringPtr name = m_session.GetOutputNameAllocated(output_index, allocator);
		Ort::TypeInfo type_info = m_session.GetOutputTypeInfo(output_index);
		Ort::ConstTensorTypeAndShapeInfo tensor_info = type_info.GetTensorTypeAndShapeInfo();
		ONNXTensorElementDataType element_type = tensor_info.GetElementType();
		std::vector<int64_t> shape = tensor_info.GetShape();
		m_outputs.emplace_back(std::string(name.get()), shape.size(), shape, element_type);
	}
}

Ort::Session InferenceSession::create_session(const char *model_path,
	InferenceSessionProvider provider,
	int intra_op_num_threads)
{
	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(intra_op_num_threads);
	return Ort::Session(m_ort_env, model_path, session_options);
}

std::vector<InferenceSession::SessionInputOutput> InferenceSession::get_inputs()
{
	return m_inputs;
}

std::vector<InferenceSession::SessionInputOutput> InferenceSession::get_outputs()
{
	return m_outputs;
}

void InferenceSession::validate_input(const std::vector<nb::ndarray<nb::numpy>> &inputs)
{
	if (inputs.size() != m_inputs.size()) {
		throw std::runtime_error(
			std::format(
				"Wrong Input Dimensions, expected: {}, got: {}",
				m_inputs.size(),
				inputs.size()));
	}

	for (size_t i = 0; i < inputs.size(); i++) {
		const auto &input = inputs[i];
		if (input.device_type() != nb::device::cpu::value) {
			throw std::runtime_error("Input arrays must be cpu only!");
		}

		const auto &expected_dtype = m_dtype_conversion_map.at(m_inputs[i].type);
		if (input.dtype() != expected_dtype) {
			throw std::runtime_error(std::format("Wrong dtype for input {}", i));
		}

		if (input.ndim() != m_inputs[i].rank) {
			throw std::runtime_error(
				std::format("Wrong ndim for input expected: {}, got: {}",
					m_inputs[i].rank, input.ndim()));
		}

		// contiguity check
		if (input.ndim() > 0 && input.size() > 1) {
			bool continguous = true;
			int64_t accum = 1;
			for (int64_t j = input.ndim() - 1; j >= 0; j--) {
				if (!continguous)
					break;
				continguous = input.shape(j) == 1 || input.stride(j) == accum;
				accum *= static_cast<int64_t>(input.shape(j));
			}
			if (!continguous) {
				throw std::invalid_argument(std::format("input {} must be C-contiguous", i));
			}
		}
	}
}
std::vector<nb::ndarray<nb::numpy>> InferenceSession::run(const std::vector<nb::ndarray<nb::numpy>> &inputs)
{
	validate_input(inputs);

	std::vector<Ort::Value> input_tensors;
	for (size_t i = 0; i < inputs.size(); i++) {
		const auto &input = inputs[i];
		std::vector<int64_t> input_shape(input.ndim());
		for (size_t j = 0; j < input.ndim(); j++) {
			input_shape[j] = static_cast<int64_t>(input.shape(j));
		}

		input_tensors.push_back(
			Ort::Value::CreateTensor(m_memory_info,
				input.data(),
				input.nbytes(),
				input_shape.data(),
				input_shape.size(),
				m_inputs[i].type));
	}
	std::vector<const char *> in_names, out_names;
	for (const auto &input : m_inputs) {
		in_names.push_back(input.name.c_str());
	}
	for (const auto &output : m_outputs) {
		out_names.push_back(output.name.c_str());
	}

	std::vector<Ort::Value> output_tensors;
	{
		nb::gil_scoped_release release;
		output_tensors = m_session.Run(
			Ort::RunOptions { }, in_names.data(), input_tensors.data(),
			input_tensors.size(), out_names.data(), out_names.size());
	}

	std::vector<nb::ndarray<nb::numpy>> outputs;
	outputs.reserve(output_tensors.size());
	for (size_t i = 0; i < output_tensors.size(); i++) {
		const auto &output_tensor = output_tensors[i];
		auto info = output_tensor.GetTensorTypeAndShapeInfo();

		const auto type = info.GetElementType();
		const auto shape = info.GetShape();
		const auto dtype = m_dtype_conversion_map.at(type);
		const size_t nbytes = info.GetElementCount() * (dtype.bits / 8);

		uint8_t *buffer = new uint8_t[nbytes];
		std::memcpy(buffer, output_tensor.GetTensorRawData(), nbytes);

		nb::capsule owner(buffer, [](void *p) noexcept {
			delete[] static_cast<uint8_t *>(p);
		});

		std::vector<size_t> s_shape;
		s_shape.reserve(shape.size());
		for (int64_t dim : shape)
			s_shape.push_back(static_cast<size_t>(dim));

		outputs.push_back(
			nb::ndarray<nb::numpy>(buffer, s_shape.size(), s_shape.data(), owner, { }, dtype));
	}

	return outputs;
}
