#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace cpp_onnx_python {

// ---------------------------------------------------------------------------
// Errors. The binding maps these onto Python built-ins of the same name.
// ---------------------------------------------------------------------------

class Error : public std::runtime_error {
public:
	explicit Error(const std::string &what)
		: std::runtime_error(what)
	{
	}
};

class ValueError : public Error {
	using Error::Error;
};
class FileNotFoundError : public Error {
	using Error::Error;
};
class NotImplementedError : public Error {
	using Error::Error;
};
class RuntimeError : public Error {
	using Error::Error;
};

// ---------------------------------------------------------------------------
// Tensor element types. Values mirror ONNXTensorElementDataType.
// ---------------------------------------------------------------------------

enum class DataType : int32_t {
	Undefined = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED,
	Float = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
	UInt8 = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
	Int8 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
	UInt16 = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
	Int16 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
	Int32 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
	Int64 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
	String = ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
	Bool = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
	Float16 = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
	Double = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
	UInt32 = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
	UInt64 = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
	Complex64 = ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64,
	Complex128 = ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128,
	BFloat16 = ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
};

/// Both tables are indexed by the ONNX enum value, which is contiguous 0..16.
inline std::size_t element_size(DataType type)
{
	static constexpr std::size_t sizes[] = {
		0, 4, 1, 1, 2, 2, 4, 8, 0, 1, 2, 8, 4, 8, 8, 16, 2,
	};
	const auto index = static_cast<std::size_t>(type);
	return index < std::size(sizes) ? sizes[index] : 0;
}

inline const char *data_type_name(DataType type)
{
	static constexpr const char *names[] = {
		"undefined", "float32", "uint8", "int8", "uint16", "int16",
		"int32", "int64", "string", "bool", "float16", "float64",
		"uint32", "uint64", "complex64", "complex128", "bfloat16",
	};
	const auto index = static_cast<std::size_t>(type);
	return index < std::size(names) ? names[index] : "undefined";
}

/// True when the core can move this type in and out of Python. `String` and
/// `BFloat16` are rejected rather than exposed as packed bytes / fp16-as-uint16.
inline bool is_numpy_compatible(DataType type)
{
	return type != DataType::BFloat16 && element_size(type) > 0;
}

/// Format a shape for error messages; dynamic dimensions become `?`.
inline std::string format_shape(const int64_t *shape, std::size_t rank)
{
	std::string result = "(";
	for (std::size_t i = 0; i < rank; ++i) {
		result += i ? ", " : "";
		result += shape[i] < 0 ? "?" : std::to_string(shape[i]);
	}
	return result + ")";
}

// ---------------------------------------------------------------------------
// Session API.
// ---------------------------------------------------------------------------

/// CPU is always available; GPU selects CUDA on Linux/Windows and CoreML on
/// macOS, and fails loudly when the linked build does not expose the provider.
enum class Provider {
	CPU,
	GPU,
};

enum class OptimizationLevel {
	Disable,
	Basic,
	Extended,
	All,
};

/// Static description of one model input or output.
struct TensorInfo {
	std::string name;
	DataType type = DataType::Undefined;
	/// Element dimensions; `-1` marks a dynamic (symbolic) dimension.
	std::vector<int64_t> shape;

	std::size_t rank() const noexcept { return shape.size(); }
};

/// Borrowed input view. `data` and `shape` must stay valid for the duration of
/// `run`; the shape is not copied so a NumPy array's own buffer is passed
/// straight through to ONNX Runtime.
struct InputTensor {
	const void *data = nullptr;
	DataType type = DataType::Undefined;
	const int64_t *shape = nullptr;
	std::size_t rank = 0;
	std::size_t nbytes = 0;
};

/// Owned output; the buffer is released into Python as a capsule.
struct OutputTensor {
	std::unique_ptr<uint8_t[]> data;
	std::size_t nbytes = 0;
	DataType type = DataType::Undefined;
	std::vector<int64_t> shape;
};

struct SessionOptions {
	/// `0` keeps the ONNX Runtime default.
	int intra_op_num_threads = 0;
	/// `0` keeps the ONNX Runtime default; `> 0` enables parallel execution.
	int inter_op_num_threads = 0;
	OptimizationLevel graph_optimization_level = OptimizationLevel::All;
	/// Raw provider options for the GPU execution provider.
	std::unordered_map<std::string, std::string> provider_options;
	/// Raw `AddConfigEntry` keys/values for `Ort::SessionOptions`.
	std::unordered_map<std::string, std::string> session_config;
};

/// Owns one ONNX Runtime session and its cached metadata. Deliberately free of
/// any Python dependency so the native tests and benchmarks exercise the same
/// code as the binding.
class InferenceSession {
public:
	/// \throws FileNotFoundError, ValueError, RuntimeError.
	explicit InferenceSession(std::filesystem::path model_path,
		Provider provider = Provider::CPU,
		SessionOptions options = {});

	const std::vector<TensorInfo> &inputs() const noexcept { return m_inputs; }
	const std::vector<TensorInfo> &outputs() const noexcept { return m_outputs; }

	/// \throws ValueError, NotImplementedError, RuntimeError.
	std::vector<OutputTensor> run(const std::vector<InputTensor> &inputs);

private:
	Ort::Session create_session(const std::filesystem::path &model_path,
		Provider provider,
		const SessionOptions &options);
	void inspect_model();
	void validate_inputs(const std::vector<InputTensor> &inputs) const;

	Ort::MemoryInfo m_memory_info;
	/// Created once: re-creating run options on every call allocates inside ORT.
	Ort::RunOptions m_run_options;
	Ort::Session m_session { nullptr };
	std::vector<TensorInfo> m_inputs;
	std::vector<TensorInfo> m_outputs;
	/// Cached `TensorInfo::name.c_str()` views for the run hot path.
	std::vector<const char *> m_input_names;
	std::vector<const char *> m_output_names;
};

} // namespace cpp_onnx_python
