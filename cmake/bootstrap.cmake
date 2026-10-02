cmake_minimum_required(VERSION 3.15)

set(DEPS_DIR ${CMAKE_SOURCE_DIR}/third-party/)
set(ORT_DIR "${DEPS_DIR}")
set(ORT_FINAL_DIR "${ORT_DIR}/onnxruntime")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    set(ORT_ARCH "arm64")
else()
    set(ORT_ARCH "x64")
endif()
if(WIN32)
    if (ORT_ARCH STREQUAL "arm64")
        set(ORT_ARCHIVE "onnxruntime-win-arm64x-1.30.0.zip")
    else()
        set(ORT_ARCHIVE "onnxruntime-win-x64-gpu_cuda12-1.30.0.zip")
    endif()
elseif(APPLE)
    if (ORT_ARCH STREQUAL "arm64")
        set(ORT_ARCHIVE "onnxruntime-osx-arm64-1.30.0.tgz")
    else()
        message(FATAL_ERROR "Unsupported platform")
    endif()
elseif(UNIX)
    if (ORT_ARCH STREQUAL "arm64")
        set(ORT_ARCHIVE "onnxruntime-linux-aarch64-1.30.0.tgz")
    else()
        set(ORT_ARCHIVE "onnxruntime-linux-x64-gpu_cuda12-1.30.0.tgz")
    endif()
else()
    message(FATAL_ERROR "Unsupported platform")
endif()


set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.30.0/${ORT_ARCHIVE}")
set(ARCHIVE_PATH "${CMAKE_BINARY_DIR}/${ORT_ARCHIVE}")

if(EXISTS ${ARCHIVE_PATH})
    message(STATUS "ORT already downloaded")
else()
    file(
        DOWNLOAD ${ORT_URL} ${ARCHIVE_PATH}
        SHOW_PROGRESS
    )
endif()
file(
    ARCHIVE_EXTRACT
    INPUT ${CMAKE_BINARY_DIR}/${ORT_ARCHIVE}
    DESTINATION ${ORT_DIR}
)
file(
    GLOB ORT_EXTRACTED_DIR_LIST
    LIST_DIRECTORIES true
    "${ORT_DIR}/onnxruntime-*"
)
list(
    GET ORT_EXTRACTED_DIR_LIST 0 ORT_EXTRACTED_DIR
)
if(EXISTS ${ORT_FINAL_DIR})
    file(
        REMOVE_RECURSE ${ORT_FINAL_DIR}
    )
endif()
file(
    RENAME ${ORT_EXTRACTED_DIR} ${ORT_FINAL_DIR}
)

if (WIN32)
    set(ORT_IMPLIB "${ORT_FINAL_DIR}/lib/onnxruntime.lib")
    set(ORT_RUNTIME_DLL "${ORT_FINAL_DIR}/lib/onnxruntime.dll")
else()
    set(ORT_LIB_FILE "${ORT_FINAL_DIR}/lib/libonnxruntime.so")
    if (APPLE)
        set(ORT_LIB_FILE "${ORT_FINAL_DIR}/lib/libonnxruntime.dylib")
    endif()
endif()

add_library(onnxruntime SHARED IMPORTED GLOBAL)
if(WIN32)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_IMPLIB "${ORT_IMPLIB}"
        IMPORTED_LOCATION "${ORT_RUNTIME_DLL}")
else()
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ORT_LIB_FILE}")
endif()
set_target_properties(onnxruntime PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_FINAL_DIR}/include")
