cmake_minimum_required(VERSION 3.15...3.26)

project(cpp-onnx-python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

nanobind_add_module(
  cpp_onnx_python_ext
  STABLE_ABI
  NB_STATIC
  src/cpp_onnx_python_ext.cpp
  src/inference_session.cpp
)

target_link_libraries(cpp_onnx_python_ext PRIVATE onnxruntime)
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

# need to add runtime destination so that windows works
# library for macos / linux
install(TARGETS cpp_onnx_python_ext
    LIBRARY DESTINATION cpp_onnx_python
    RUNTIME DESTINATION cpp_onnx_python)

if(WIN32)
    install(FILES "${ORT_RUNTIME_DLL}" DESTINATION cpp_onnx_python)
else()
    # Real file plus the SONAME / install-name symlinks, resolved
    # relative to lib/.
    install(DIRECTORY "${ORT_DIR}/lib/"
        DESTINATION cpp_onnx_python
        FILES_MATCHING
        PATTERN "${ORT_LIB_NAME}*"
        PATTERN "*.dSYM" EXCLUDE
        PATTERN "cmake" EXCLUDE
        PATTERN "pkgconfig" EXCLUDE)
endif()
