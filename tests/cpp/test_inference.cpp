// Minimal, dependency-free native test driver for the cpp_onnx_python core.
//
// Built and run via CTest for plain CMake builds:
//   cmake -S . -B build-native
//   cmake --build build-native
//   ctest --test-dir build-native --output-on-failure

#include "cpp_onnx_python/inference.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace cpp_onnx_python;

namespace {

const char *kFixtureDir = CPP_ONNX_PYTHON_FIXTURE_DIR;
constexpr float kScale = 0.5f;
constexpr float kBias = 1.0f;

int g_failures = 0;
int g_checks = 0;

#define CHECK(condition)                                                        \
	do {                                                                     \
		++g_checks;                                                          \
		if (!(condition)) {                                                    \
			++g_failures;                                                    \
			std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": "      \
				  << #condition << "\n";                                 \
		}                                                                    \
	} while (false)

#define CHECK_THROWS(expression, ExceptionType)                                 \
	do {                                                                     \
		++g_checks;                                                          \
		bool threw_expected = false;                                         \
		try {                                                                \
			(void)(expression);                                             \
		} catch (const ExceptionType &) {                                    \
			threw_expected = true;                                           \
		} catch (const std::exception &e) {                                  \
			std::cerr << "FAIL " << __FILE__ << ":" << __LINE__             \
				  << ": wrong exception: " << e.what() << "\n";          \
		}                                                                    \
		if (!threw_expected) {                                               \
			++g_failures;                                                    \
			std::cerr << "FAIL " << __FILE__ << ":" << __LINE__             \
				  << ": expected " #ExceptionType " from " #expression   \
				  << "\n";                                              \
		}                                                                    \
	} while (false)

std::string fixture(const char *name)
{
	return std::string(kFixtureDir) + "/" + name;
}

const std::vector<int64_t> kShape { 1, 3, 8, 8 };
const std::vector<int64_t> kShapeRank3 { 1, 3, 8 };
const std::vector<int64_t> kShapeWrong { 1, 3, 8, 7 };

std::vector<float> make_input_data()
{
	std::vector<float> data(1 * 3 * 8 * 8);
	for (std::size_t i = 0; i < data.size(); ++i)
		data[i] = std::sin(static_cast<float>(i) * 0.37f) * 2.0f;
	return data;
}

InputTensor make_input(const std::vector<float> &data,
	const std::vector<int64_t> &shape)
{
	InputTensor input;
	input.data = data.data();
	input.type = DataType::Float;
	input.shape = shape.data();
	input.rank = shape.size();
	input.nbytes = data.size() * sizeof(float);
	return input;
}

InferenceSession make_session()
{
	return InferenceSession(fixture("mul_add.onnx"), Provider::CPU, SessionOptions {});
}

void test_metadata()
{
	InferenceSession session = make_session();

	CHECK(session.inputs().size() == 1);
	CHECK(session.outputs().size() == 1);
	if (!session.inputs().empty()) {
		const TensorInfo &input = session.inputs()[0];
		CHECK(input.name == "X");
		CHECK(input.type == DataType::Float);
		CHECK(input.rank() == 4);
		CHECK((input.shape == std::vector<int64_t> { 1, 3, 8, 8 }));
	}
	if (!session.outputs().empty()) {
		const TensorInfo &output = session.outputs()[0];
		CHECK(output.name == "Y");
		CHECK(output.type == DataType::Float);
		CHECK((output.shape == std::vector<int64_t> { 1, 3, 8, 8 }));
	}
}

void test_run_values()
{
	InferenceSession session = make_session();
	const std::vector<float> data = make_input_data();
	const InputTensor input = make_input(data, kShape);

	const std::vector<OutputTensor> outputs = session.run({ input });
	CHECK(outputs.size() == 1);
	if (outputs.empty())
		return;

	const OutputTensor &output = outputs[0];
	CHECK(output.type == DataType::Float);
	CHECK((output.shape == std::vector<int64_t> { 1, 3, 8, 8 }));
	CHECK(output.nbytes == data.size() * sizeof(float));
	if (output.nbytes != data.size() * sizeof(float))
		return;

	const float *values = reinterpret_cast<const float *>(output.data.get());
	float max_error = 0.0f;
	for (std::size_t i = 0; i < data.size(); ++i) {
		const float expected = data[i] * kScale + kBias;
		max_error = std::max(max_error, std::fabs(values[i] - expected));
	}
	CHECK(max_error < 1e-6f);
}

void test_repeated_calls()
{
	InferenceSession session = make_session();
	const std::vector<float> data = make_input_data();
	const InputTensor input = make_input(data, kShape);

	const std::vector<OutputTensor> first = session.run({ input });
	const std::vector<OutputTensor> second = session.run({ input });
	CHECK(first.size() == 1 && second.size() == 1);
	if (first.empty() || second.empty())
		return;
	CHECK(std::memcmp(first[0].data.get(), second[0].data.get(), first[0].nbytes) == 0);
}

void test_two_sessions()
{
	InferenceSession a = make_session();
	InferenceSession b = make_session();
	CHECK(a.inputs().size() == b.inputs().size());
	const std::vector<float> data = make_input_data();
	const InputTensor input = make_input(data, kShape);
	CHECK(a.run({ input }).size() == 1);
	CHECK(b.run({ input }).size() == 1);
}

void test_input_validation()
{
	InferenceSession session = make_session();
	const std::vector<float> data = make_input_data();

	InputTensor wrong_dtype = make_input(data, kShape);
	wrong_dtype.type = DataType::Double;
	CHECK_THROWS(session.run({ wrong_dtype }), ValueError);

	InputTensor wrong_rank = make_input(data, kShapeRank3);
	CHECK_THROWS(session.run({ wrong_rank }), ValueError);

	InputTensor wrong_shape = make_input(data, kShapeWrong);
	CHECK_THROWS(session.run({ wrong_shape }), ValueError);

	InputTensor too_small = make_input(data, kShape);
	too_small.nbytes = sizeof(float);
	CHECK_THROWS(session.run({ too_small }), ValueError);

	CHECK_THROWS(session.run({}), ValueError);
	CHECK_THROWS(session.run({ make_input(data, kShape), make_input(data, kShape) }),
		ValueError);
}

void test_constructor_validation()
{
	CHECK_THROWS(InferenceSession(fixture("does-not-exist.onnx")), FileNotFoundError);
	CHECK_THROWS(InferenceSession(""), ValueError);

	SessionOptions negative_threads;
	negative_threads.intra_op_num_threads = -1;
	CHECK_THROWS(InferenceSession(fixture("mul_add.onnx"), Provider::CPU, negative_threads),
		ValueError);

	SessionOptions negative_inter;
	negative_inter.inter_op_num_threads = -2;
	CHECK_THROWS(InferenceSession(fixture("mul_add.onnx"), Provider::CPU, negative_inter),
		ValueError);
}

void test_session_options()
{
	SessionOptions options;
	options.intra_op_num_threads = 1;
	options.inter_op_num_threads = 1;
	options.graph_optimization_level = OptimizationLevel::Basic;
	options.session_config["session.disable_cpu_ep_fallback"] = "0";

	InferenceSession session(fixture("mul_add.onnx"), Provider::CPU, options);
	const std::vector<float> data = make_input_data();
	CHECK(session.run({ make_input(data, kShape) }).size() == 1);
}

} // namespace

int main()
{
	const std::vector<std::pair<std::string, std::function<void()>>> tests = {
		{ "metadata", test_metadata },
		{ "run_values", test_run_values },
		{ "repeated_calls", test_repeated_calls },
		{ "two_sessions", test_two_sessions },
		{ "input_validation", test_input_validation },
		{ "constructor_validation", test_constructor_validation },
		{ "session_options", test_session_options },
	};

	for (const auto &[name, test] : tests) {
		const int before = g_failures;
		std::cout << "[ RUN      ] " << name << "\n";
		try {
			test();
		} catch (const std::exception &e) {
			++g_failures;
			std::cerr << "FAIL uncaught exception in " << name << ": " << e.what()
				  << "\n";
		}
		std::cout << (g_failures == before ? "[       OK ] " : "[  FAILED  ] ")
			  << name << "\n";
	}

	std::cout << (g_failures == 0 ? "PASSED" : "FAILED") << ": " << g_checks
		  << " checks, " << g_failures << " failure(s)\n";
	return g_failures == 0 ? 0 : 1;
}
