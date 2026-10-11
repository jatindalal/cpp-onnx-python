cmake_minimum_required(VERSION 3.26)

project(cpp-onnx-python LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT SKBUILD)
  message(WARNING "\
  compile the package in an isolated environment, and then install it.
  This CMake file is usually used using 'scikit-build'
  =====================================================================
   $ pip install .
  =====================================================================
  if a development instance use
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
   or
   $ pip install --no-build-isolation -ve -Ceditable.rebuild=true .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to re-run the above
  after editing C++ files.")
endif()

find_package(Python 3.12
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

if(NOT SKBUILD)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE NANOBIND_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  message(STATUS "nanobind dir: ${NANOBIND_CMAKE_DIR}")
  list(APPEND CMAKE_PREFIX_PATH "${NANOBIND_CMAKE_DIR}")
endif()

find_package(nanobind CONFIG REQUIRED)

include(cmake/bootstrap.cmake)

# Python-free core: exercised directly by the native tests and the C++ example.
add_library(cpp_onnx_python_core STATIC src/inference_session.cpp)
target_include_directories(cpp_onnx_python_core PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(cpp_onnx_python_core PUBLIC onnxruntime)
set_target_properties(cpp_onnx_python_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

nanobind_add_module(
  cpp_onnx_python_ext
  STABLE_ABI
  NB_STATIC
  src/cpp_onnx_python_ext.cpp
)

target_compile_definitions(cpp_onnx_python_ext PRIVATE
  CPP_ONNX_PYTHON_VERSION="${PROJECT_VERSION}")

target_link_libraries(cpp_onnx_python_ext PRIVATE cpp_onnx_python_core)
target_include_directories(cpp_onnx_python_ext PRIVATE ${CMAKE_SOURCE_DIR}/include)

if(APPLE)
    set(EXT_RPATH "@loader_path")
elseif(UNIX)
    set(EXT_RPATH "$ORIGIN")
endif()
if(EXT_RPATH)
    set_property(TARGET cpp_onnx_python_ext PROPERTY BUILD_RPATH "${EXT_RPATH}")
    set_property(TARGET cpp_onnx_python_ext PROPERTY INSTALL_RPATH "${EXT_RPATH}")
endif()

# Native test + benchmark: built for plain CMake builds, skipped in wheel builds.
if(NOT SKBUILD)
    enable_testing()
    add_executable(cpp_onnx_python_tests tests/cpp/test_inference.cpp)
    target_link_libraries(cpp_onnx_python_tests PRIVATE cpp_onnx_python_core)
    target_compile_definitions(cpp_onnx_python_tests PRIVATE
        CPP_ONNX_PYTHON_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures")
    add_test(NAME cpp_onnx_python_tests COMMAND cpp_onnx_python_tests)

    add_executable(cpp_onnx_python_benchmark benchmarks/cpp_benchmark.cpp)
    target_link_libraries(cpp_onnx_python_benchmark PRIVATE cpp_onnx_python_core)
endif()

# need to add runtime destination so that windows works
# library for macos / linux
install(TARGETS cpp_onnx_python_ext
    LIBRARY DESTINATION cpp_onnx_python
    RUNTIME DESTINATION cpp_onnx_python)

if(WIN32)
    install(FILES "${ORT_RUNTIME_DLL}" DESTINATION cpp_onnx_python)
else()
    # Real library plus the SONAME / install-name symlinks. The unversioned
    # development symlink is not needed at runtime and is excluded to avoid
    # shipping a second copy of the library.
    install(DIRECTORY "${ORT_FINAL_DIR}/lib/"
        DESTINATION cpp_onnx_python
        FILES_MATCHING
        PATTERN "${ORT_LIB_NAME}*"
        PATTERN "${ORT_LIB_NAME}.dylib" EXCLUDE
        PATTERN "${ORT_LIB_NAME}.so" EXCLUDE
        PATTERN "*.dSYM" EXCLUDE
        PATTERN "cmake" EXCLUDE
        PATTERN "pkgconfig" EXCLUDE)
endif()

# Third-party notices required for redistribution.
install(FILES
    "${CMAKE_SOURCE_DIR}/LICENSE"
    DESTINATION cpp_onnx_python/licenses)
install(FILES
    "${CMAKE_SOURCE_DIR}/licenses/onnxruntime-LICENSE.txt"
    "${CMAKE_SOURCE_DIR}/licenses/onnxruntime-ThirdPartyNotices.txt"
    DESTINATION cpp_onnx_python/licenses)

# Tiny deterministic demo model, so `demo_model_path()` makes the installed
# wheel runnable without any extra download.
install(FILES "${CMAKE_SOURCE_DIR}/tests/fixtures/mul_add.onnx"
    DESTINATION cpp_onnx_python/_demo)
