cmake_minimum_required(VERSION 3.15...3.26)

project(cpp-onnx-python LANGUAGES CXX)

if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build'. Running
  it directly will almost certainly not produce the desired result. If
  you are a user trying to install this package, please use the command
  below, which will install all necessary build dependencies, compile
  the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to re-run the above
  after editing C++ files.")
endif()

find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

if(NOT SKBUILD)
  execute_process(
    COMMAND python -m nanobind --cmake_dir
    OUTPUT_VARIABLE NANOBIND_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  message(STATUS "nanobind dir: ${NANOBIND_CMAKE_DIR}")
  list(APPEND CMAKE_PREFIX_PATH "${NANOBIND_CMAKE_DIR}")
endif()

find_package(nanobind CONFIG REQUIRED)

include(cmake/bootstrap.cmake)

add_library(onnxruntime SHARED IMPORTED GLOBAL)

if(WIN32)
    set(LIB_LOCATION "${ORT_DIR}/lib/onnxruntime.dll")
elseif(APPLE)
    set(LIB_LOCATION "${ORT_DIR}/lib/libonnxruntime.dylib")
elseif(UNIX)
    set(LIB_LOCATION "${ORT_DIR}/lib/libonnxruntime.so")
endif()
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${LIB_LOCATION}"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_DIR}/include")

nanobind_add_module(
  cpp_onnx_python_ext
  STABLE_ABI
  NB_STATIC
  src/cpp_onnx_python_ext.cpp
)

target_link_libraries(cpp_onnx_python_ext PRIVATE onnxruntime)

# Install directive for scikit-build-core
install(TARGETS cpp_onnx_python_ext LIBRARY DESTINATION cpp_onnx_python)
