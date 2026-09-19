#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>

namespace nb = nanobind;

using namespace nb::literals;

nb::dict model_info(const char *model_path)
{
	nb::dict info;
	nb::list input_info, output_info;
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "model info");
	Ort::SessionOptions session_options;
	session_options.SetInterOpNumThreads(0);
	session_options.SetGraphOptimizationLevel(
		GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
	Ort::Session session(env, model_path, session_options);
	Ort::AllocatorWithDefaultOptions allocator;
	size_t input_count = session.GetInputCount();
	size_t output_count = session.GetOutputCount();
	for (size_t input_index = 0; input_index < input_count; input_index++) {
		Ort::AllocatedStringPtr name = session.GetInputNameAllocated(input_index, allocator);
		Ort::TypeInfo type_info = session.GetInputTypeInfo(input_index);
		Ort::ConstTensorTypeAndShapeInfo tensor_info = type_info.GetTensorTypeAndShapeInfo();
		ONNXTensorElementDataType element_type = tensor_info.GetElementType();
		std::vector<int64_t> shape = tensor_info.GetShape();

		nb::dict info;
		info["name"] = name.get();
		info["rank"] = shape.size();
		info["shape"] = shape;
		info["type"] = element_type;

		input_info.append(info);
	}
	for (size_t output_index = 0; output_index < output_count; output_index++) {
		Ort::AllocatedStringPtr name = session.GetOutputNameAllocated(output_index, allocator);
		Ort::TypeInfo type_info = session.GetOutputTypeInfo(output_index);
		Ort::ConstTensorTypeAndShapeInfo tensor_info = type_info.GetTensorTypeAndShapeInfo();
		ONNXTensorElementDataType element_type = tensor_info.GetElementType();
		std::vector<int64_t> shape = tensor_info.GetShape();

		nb::dict info;
		info["name"] = name.get();
		info["rank"] = shape.size();
		info["shape"] = shape;
		info["type"] = element_type;

		output_info.append(info);
	}

	info["input"] = input_info;
	info["output"] = output_info;
	return info;
}

NB_MODULE(cpp_onnx_python_ext, m)
{
	nb::enum_<ONNXTensorElementDataType>(m, "ONNXTensorElementDatatype")
		.value("UNDEFINED", ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED)
		.value("FLOAT", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
		.value("UINT8", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)
		.value("INT8", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8)
		.value("UINT16", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16)
		.value("INT16", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16)
		.value("INT32", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
		.value("INT64", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
		.value("STRING", ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING)
		.value("BOOL", ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL)
		.value("FLOAT16", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
		.value("DOUBLE", ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)
		.value("UINT32", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32)
		.value("UINT64", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64)
		.value("COMPLEX64", ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64)
		.value("COMPLEX128", ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128)
		.value("BFLOAT16", ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);

	m.doc() = "Onnxruntime bindings";
	m.def("model_info", &model_info, nb::arg("model_path"));
}
