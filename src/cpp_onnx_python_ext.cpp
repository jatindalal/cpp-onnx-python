#include "cpp_onnx_python/inference.hpp"

#include <exception>
#include <filesystem>
#include <format>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
namespace cop = cpp_onnx_python;

namespace {

/// NumPy <-> DataType mapping, in one table used by both directions.
struct NumpyDtype {
	nb::dlpack::dtype_code code;
	uint8_t bits;
	cop::DataType type;
};

constexpr NumpyDtype kDtypes[] = {
	{ nb::dlpack::dtype_code::Float, 16, cop::DataType::Float16 },
	{ nb::dlpack::dtype_code::Float, 32, cop::DataType::Float },
	{ nb::dlpack::dtype_code::Float, 64, cop::DataType::Double },
	{ nb::dlpack::dtype_code::Int, 8, cop::DataType::Int8 },
	{ nb::dlpack::dtype_code::Int, 16, cop::DataType::Int16 },
	{ nb::dlpack::dtype_code::Int, 32, cop::DataType::Int32 },
	{ nb::dlpack::dtype_code::Int, 64, cop::DataType::Int64 },
	{ nb::dlpack::dtype_code::UInt, 8, cop::DataType::UInt8 },
	{ nb::dlpack::dtype_code::UInt, 16, cop::DataType::UInt16 },
	{ nb::dlpack::dtype_code::UInt, 32, cop::DataType::UInt32 },
	{ nb::dlpack::dtype_code::UInt, 64, cop::DataType::UInt64 },
	{ nb::dlpack::dtype_code::Bool, 8, cop::DataType::Bool },
	{ nb::dlpack::dtype_code::Complex, 64, cop::DataType::Complex64 },
	{ nb::dlpack::dtype_code::Complex, 128, cop::DataType::Complex128 },
};

nb::dlpack::dtype to_dlpack(cop::DataType type)
{
	for (const NumpyDtype &entry : kDtypes)
		if (entry.type == type)
			return { static_cast<uint8_t>(entry.code), entry.bits, 1 };
	throw cop::NotImplementedError(
		std::format("Cannot expose '{}' to NumPy", cop::data_type_name(type)));
}

cop::DataType from_dlpack(const nb::dlpack::dtype &dtype)
{
	for (const NumpyDtype &entry : kDtypes)
		if (static_cast<uint8_t>(entry.code) == dtype.code &&
			entry.bits == dtype.bits)
			return entry.type;
	throw cop::NotImplementedError(
		std::format("Unsupported NumPy dtype (code={}, bits={})", dtype.code, dtype.bits));
}

bool is_c_contiguous(const nb::ndarray<nb::numpy> &array)
{
	if (array.ndim() == 0 || array.size() <= 1)
		return true;
	int64_t expected_stride = 1;
	for (int64_t dim = static_cast<int64_t>(array.ndim()) - 1; dim >= 0; --dim) {
		const int64_t extent = static_cast<int64_t>(array.shape(dim));
		if (extent == 1)
			continue;
		if (array.stride(dim) != expected_stride)
			return false;
		expected_stride *= extent;
	}
	return true;
}

cop::InputTensor to_input_tensor(const nb::ndarray<nb::numpy> &array, std::size_t index)
{
	if (array.device_type() != nb::device::cpu::value)
		throw cop::ValueError(std::format("Input {} must be a CPU array", index));
	if (!is_c_contiguous(array))
		throw cop::ValueError(std::format(
			"Input {} must be C-contiguous; call numpy.ascontiguousarray() first",
			index));

	// `from_dlpack` rejects every type the binding cannot represent, and the
	// shape buffer stays alive for the duration of run().
	cop::InputTensor tensor;
	tensor.data = array.data();
	tensor.type = from_dlpack(array.dtype());
	tensor.nbytes = array.nbytes();
	tensor.shape = array.shape_ptr();
	tensor.rank = array.ndim();
	return tensor;
}

cop::OptimizationLevel parse_optimization_level(const std::string &value)
{
	if (value == "disable" || value == "none")
		return cop::OptimizationLevel::Disable;
	if (value == "basic")
		return cop::OptimizationLevel::Basic;
	if (value == "extended")
		return cop::OptimizationLevel::Extended;
	if (value == "all")
		return cop::OptimizationLevel::All;
	throw cop::ValueError(std::format(
		"graph_optimization_level must be one of 'disable', 'basic', 'extended', "
		"'all', got '{}'",
		value));
}

} // namespace

NB_MODULE(cpp_onnx_python_ext, m)
{
	m.doc() = "ONNX Runtime inference bindings for cpp-onnx-python";
	m.attr("__version__") = CPP_ONNX_PYTHON_VERSION;

	// Map native exceptions onto Python built-ins: misuse (ValueError), missing
	// models (FileNotFoundError), unsupported features (NotImplementedError)
	// and runtime failures (RuntimeError). Rethrow anything else so nanobind's
	// default translator still runs.
	nb::register_exception_translator([](const std::exception_ptr &ptr, void *) {
		try {
			std::rethrow_exception(ptr);
		} catch (const cop::FileNotFoundError &e) {
			PyErr_SetString(PyExc_FileNotFoundError, e.what());
		} catch (const cop::NotImplementedError &e) {
			PyErr_SetString(PyExc_NotImplementedError, e.what());
		} catch (const cop::ValueError &e) {
			PyErr_SetString(PyExc_ValueError, e.what());
		} catch (const cop::RuntimeError &e) {
			PyErr_SetString(PyExc_RuntimeError, e.what());
		} catch (...) {
			throw;
		}
	});

	nb::enum_<cop::DataType>(m, "DataType")
		.value("UNDEFINED", cop::DataType::Undefined)
		.value("FLOAT", cop::DataType::Float)
		.value("UINT8", cop::DataType::UInt8)
		.value("INT8", cop::DataType::Int8)
		.value("UINT16", cop::DataType::UInt16)
		.value("INT16", cop::DataType::Int16)
		.value("INT32", cop::DataType::Int32)
		.value("INT64", cop::DataType::Int64)
		.value("STRING", cop::DataType::String)
		.value("BOOL", cop::DataType::Bool)
		.value("FLOAT16", cop::DataType::Float16)
		.value("DOUBLE", cop::DataType::Double)
		.value("UINT32", cop::DataType::UInt32)
		.value("UINT64", cop::DataType::UInt64)
		.value("COMPLEX64", cop::DataType::Complex64)
		.value("COMPLEX128", cop::DataType::Complex128)
		.value("BFLOAT16", cop::DataType::BFloat16);

	nb::enum_<cop::Provider>(m, "InferenceSessionProvider")
		.value("CPU", cop::Provider::CPU)
		.value("GPU", cop::Provider::GPU);

	nb::class_<cop::TensorInfo>(m, "TensorInfo")
		.def_ro("name", &cop::TensorInfo::name)
		.def_ro("type", &cop::TensorInfo::type)
		.def_prop_ro("shape", [](const cop::TensorInfo &self) { return self.shape; })
		.def_prop_ro("rank", [](const cop::TensorInfo &self) { return self.rank(); })
		.def_prop_ro("dtype", [](const cop::TensorInfo &self) {
			return cop::data_type_name(self.type);
		})
		.def("to_dict", [](const cop::TensorInfo &self) {
			nb::dict result;
			result["name"] = self.name;
			result["type"] = self.type;
			result["dtype"] = cop::data_type_name(self.type);
			result["shape"] = self.shape;
			result["rank"] = self.rank();
			return result;
		})
		.def("__repr__", [](const cop::TensorInfo &self) {
			return std::format("<TensorInfo name='{}' dtype={} shape={}>",
				self.name, cop::data_type_name(self.type),
				cop::format_shape(self.shape.data(), self.shape.size()));
		});

	nb::class_<cop::InferenceSession>(m, "InferenceSession")
		.def("__init__",
			[](cop::InferenceSession *self,
				std::filesystem::path model_path,
				cop::Provider provider,
				int intra_op_num_threads,
				std::unordered_map<std::string, std::string> provider_options,
				std::unordered_map<std::string, std::string> session_options,
				const std::string &graph_optimization_level,
				int inter_op_num_threads) {
				cop::SessionOptions options;
				options.intra_op_num_threads = intra_op_num_threads;
				options.inter_op_num_threads = inter_op_num_threads;
				options.provider_options = std::move(provider_options);
				options.session_config = std::move(session_options);
				options.graph_optimization_level =
					parse_optimization_level(graph_optimization_level);
				nb::gil_scoped_release release;
				new (self) cop::InferenceSession(std::move(model_path), provider,
					std::move(options));
			},
			nb::arg("model_path"),
			nb::arg("provider") = cop::Provider::CPU,
			nb::arg("intra_op_num_threads") = 0,
			nb::arg("provider_options") =
				std::unordered_map<std::string, std::string> {},
			nb::arg("session_options") =
				std::unordered_map<std::string, std::string> {},
			nb::arg("graph_optimization_level") = "all",
			nb::arg("inter_op_num_threads") = 0,
			"Load an ONNX model and create an inference session.")
		.def_prop_ro("inputs",
			[](const cop::InferenceSession &self) { return self.inputs(); },
			nb::rv_policy::copy)
		.def_prop_ro("outputs",
			[](const cop::InferenceSession &self) { return self.outputs(); },
			nb::rv_policy::copy)
		.def("run",
			[](cop::InferenceSession &self, nb::object inputs) {
				std::vector<cop::InputTensor> tensors;
				if (PyList_Check(inputs.ptr()) || PyTuple_Check(inputs.ptr())) {
					const std::size_t count = nb::len(inputs);
					tensors.reserve(count);
					for (std::size_t i = 0; i < count; ++i)
						tensors.push_back(to_input_tensor(
							nb::cast<nb::ndarray<nb::numpy>>(
								inputs[static_cast<Py_ssize_t>(i)]),
							i));
				} else {
					// A bare array is a single-input model's array.
					tensors.push_back(to_input_tensor(
						nb::cast<nb::ndarray<nb::numpy>>(inputs), 0));
				}

				std::vector<cop::OutputTensor> outputs;
				{
					nb::gil_scoped_release release;
					outputs = self.run(tensors);
				}

				nb::list result;
				for (cop::OutputTensor &output : outputs) {
					nb::capsule owner(output.data.release(),
						[](void *ptr) noexcept {
							delete[] static_cast<uint8_t *>(ptr);
						});
					// Dimensions are non-negative and, on the 64-bit
					// platforms we support, size_t and int64_t have the same
					// width, so the shape is handed over without a copy.
					static_assert(sizeof(std::size_t) == sizeof(int64_t),
						"output shape reuse requires 64-bit size_t");
					result.append(nb::ndarray<nb::numpy>(owner.data(),
						output.shape.size(),
						reinterpret_cast<const std::size_t *>(
							output.shape.data()),
						owner, { }, to_dlpack(output.type)));
				}
				return result;
			},
			nb::arg("inputs"),
			"Run inference on a bare array (single-input model) or a list/tuple "
			"of arrays, one per model input; returns a list of arrays.")
		.def("__repr__", [](const cop::InferenceSession &self) {
			return std::format("<InferenceSession inputs={} outputs={}>",
				self.inputs().size(), self.outputs().size());
		});
}
