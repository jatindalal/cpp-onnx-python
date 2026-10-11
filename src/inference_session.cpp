#include "cpp_onnx_python/inference.hpp"

#include <algorithm>
#include <cstring>
#include <format>
#include <string>
#include <utility>
#include <vector>

namespace cpp_onnx_python {
namespace {

#if defined(__APPLE__)
#define CPP_ONNX_PYTHON_GPU_PROVIDER "CoreMLExecutionProvider"
#elif defined(_WIN32) || defined(__linux__)
#define CPP_ONNX_PYTHON_GPU_PROVIDER "CUDAExecutionProvider"
#endif

const char *provider_name(Provider provider)
{
	if (provider == Provider::CPU)
		return "CPUExecutionProvider";
#ifndef CPP_ONNX_PYTHON_GPU_PROVIDER
	throw RuntimeError("GPU execution providers are not supported on this platform");
#else
	return CPP_ONNX_PYTHON_GPU_PROVIDER;
#endif
}

::GraphOptimizationLevel to_ort(OptimizationLevel level)
{
	static constexpr ::GraphOptimizationLevel levels[] = {
		ORT_DISABLE_ALL, ORT_ENABLE_BASIC, ORT_ENABLE_EXTENDED, ORT_ENABLE_ALL,
	};
	return levels[static_cast<std::size_t>(level)];
}

/// One process-wide ONNX Runtime environment; logging is registered once.
Ort::Env &shared_environment()
{
	static Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "cpp_onnx_python");
	return environment;
}

} // namespace

InferenceSession::InferenceSession(std::filesystem::path model_path,
	Provider provider,
	SessionOptions options)
	: m_memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
	if (model_path.empty())
		throw ValueError("model_path must not be empty");

	std::error_code ec;
	if (!std::filesystem::exists(model_path, ec) || ec)
		throw FileNotFoundError(
			std::format("Model file not found: '{}'", model_path.string()));
	if (!std::filesystem::is_regular_file(model_path, ec) || ec)
		throw ValueError(
			std::format("model_path is not a regular file: '{}'", model_path.string()));
	if (options.intra_op_num_threads < 0)
		throw ValueError("intra_op_num_threads must be >= 0");
	if (options.inter_op_num_threads < 0)
		throw ValueError("inter_op_num_threads must be >= 0");

	m_session = create_session(model_path, provider, options);
	inspect_model();
}

Ort::Session InferenceSession::create_session(const std::filesystem::path &model_path,
	Provider provider,
	const SessionOptions &options)
{
	const std::string requested = provider_name(provider);
	const std::vector<std::string> available = Ort::GetAvailableProviders();
	if (std::find(available.begin(), available.end(), requested) == available.end()) {
		std::string listed;
		for (const std::string &name : available)
			listed += (listed.empty() ? "" : ", ") + name;
		throw RuntimeError(std::format(
			"Provider '{}' is not available; available: {}", requested, listed));
	}

	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(options.intra_op_num_threads);
	if (options.inter_op_num_threads > 0) {
		session_options.SetExecutionMode(ORT_PARALLEL);
		session_options.SetInterOpNumThreads(options.inter_op_num_threads);
	}
	session_options.SetGraphOptimizationLevel(to_ort(options.graph_optimization_level));
	for (const auto &[key, value] : options.session_config)
		session_options.AddConfigEntry(key.c_str(), value.c_str());

	if (provider == Provider::GPU) {
#if defined(_WIN32) || defined(__linux__)
		Ort::CUDAProviderOptions cuda;
		if (!options.provider_options.empty())
			cuda.Update(options.provider_options);
		session_options.AppendExecutionProvider_CUDA_V2(*cuda);
#elif defined(__APPLE__)
		session_options.AppendExecutionProvider(requested, options.provider_options);
#endif
	}

	try {
		// path::c_str() is const char* on POSIX and const wchar_t* on Windows.
		return Ort::Session(shared_environment(), model_path.c_str(), session_options);
	} catch (const Ort::Exception &e) {
		throw RuntimeError(std::format("Failed to create a session for '{}': {}",
			model_path.string(), e.what()));
	}
}

void InferenceSession::inspect_model()
{
	Ort::AllocatorWithDefaultOptions allocator;
	const auto describe = [&](std::size_t index, bool is_input) {
		Ort::AllocatedStringPtr name = is_input
			? m_session.GetInputNameAllocated(index, allocator)
			: m_session.GetOutputNameAllocated(index, allocator);
		Ort::TypeInfo type_info = is_input
			? m_session.GetInputTypeInfo(index)
			: m_session.GetOutputTypeInfo(index);
		Ort::ConstTensorTypeAndShapeInfo info = type_info.GetTensorTypeAndShapeInfo();
		return TensorInfo {
			std::string(name.get()),
			static_cast<DataType>(info.GetElementType()),
			info.GetShape(),
		};
	};

	const std::size_t input_count = m_session.GetInputCount();
	m_inputs.reserve(input_count);
	for (std::size_t i = 0; i < input_count; ++i)
		m_inputs.push_back(describe(i, true));

	const std::size_t output_count = m_session.GetOutputCount();
	m_outputs.reserve(output_count);
	for (std::size_t i = 0; i < output_count; ++i)
		m_outputs.push_back(describe(i, false));

	for (const TensorInfo &input : m_inputs)
		m_input_names.push_back(input.name.c_str());
	for (const TensorInfo &output : m_outputs)
		m_output_names.push_back(output.name.c_str());
}

void InferenceSession::validate_inputs(const std::vector<InputTensor> &inputs) const
{
	if (inputs.size() != m_inputs.size())
		throw ValueError(std::format("Expected {} input array(s), got {}",
			m_inputs.size(), inputs.size()));

	for (std::size_t i = 0; i < inputs.size(); ++i) {
		const InputTensor &input = inputs[i];
		const TensorInfo &expected = m_inputs[i];

		if (!is_numpy_compatible(input.type))
			throw NotImplementedError(
				std::format("Input '{}' has unsupported element type '{}'",
					expected.name, data_type_name(input.type)));
		if (input.type != expected.type)
			throw ValueError(std::format("Input '{}' expects dtype '{}', got '{}'",
				expected.name, data_type_name(expected.type),
				data_type_name(input.type)));
		if (input.rank != expected.shape.size())
			throw ValueError(std::format("Input '{}' expects rank {}, got {}",
				expected.name, expected.shape.size(), input.rank));

		std::size_t element_count = 1;
		for (std::size_t dim = 0; dim < input.rank; ++dim) {
			const int64_t model_dim = expected.shape[dim];
			if (model_dim >= 0 && input.shape[dim] != model_dim)
				throw ValueError(
					std::format("Input '{}' expects shape {} but got {}",
						expected.name,
						format_shape(expected.shape.data(),
							expected.shape.size()),
						format_shape(input.shape, input.rank)));
			element_count *= static_cast<std::size_t>(
				input.shape[dim] < 0 ? 0 : input.shape[dim]);
		}

		if (input.nbytes < element_count * element_size(expected.type))
			throw ValueError(
				std::format("Input '{}' is smaller than its declared shape {}",
					expected.name, format_shape(input.shape, input.rank)));
	}
}

std::vector<OutputTensor> InferenceSession::run(const std::vector<InputTensor> &inputs)
{
	validate_inputs(inputs);

	std::vector<Ort::Value> input_tensors;
	input_tensors.reserve(inputs.size());
	for (const InputTensor &input : inputs) {
		// data/shape are borrowed for the duration of this call.
		input_tensors.push_back(Ort::Value::CreateTensor(m_memory_info,
			const_cast<void *>(input.data),
			input.nbytes,
			input.shape,
			input.rank,
			static_cast<ONNXTensorElementDataType>(input.type)));
	}

	std::vector<Ort::Value> output_tensors;
	try {
		output_tensors = m_session.Run(m_run_options,
			m_input_names.data(), input_tensors.data(), input_tensors.size(),
			m_output_names.data(), m_output_names.size());
	} catch (const Ort::Exception &e) {
		throw RuntimeError(std::format("ONNX Runtime inference failed: {}", e.what()));
	}

	std::vector<OutputTensor> outputs;
	outputs.reserve(output_tensors.size());
	for (Ort::Value &value : output_tensors) {
		const Ort::TensorTypeAndShapeInfo info = value.GetTensorTypeAndShapeInfo();
		const DataType type = static_cast<DataType>(info.GetElementType());
		if (!is_numpy_compatible(type))
			throw NotImplementedError(
				std::format("Model produced an output with unsupported element "
					    "type '{}'",
					data_type_name(type)));

		OutputTensor output;
		output.type = type;
		output.shape = info.GetShape();
		output.nbytes = info.GetElementCount() * element_size(type);
		output.data = std::make_unique<uint8_t[]>(output.nbytes);
		std::memcpy(output.data.get(), value.GetTensorRawData(), output.nbytes);
		outputs.push_back(std::move(output));
	}
	return outputs;
}

} // namespace cpp_onnx_python
