cmake_minimum_required(VERSION 3.26)

# ONNX Runtime bootstrap: downloads the official release archive into
# third-party/onnxruntime (git-ignored) and exposes the imported `onnxruntime`
# target. A second configure reuses a complete installation.

set(ORT_VERSION "1.30.0")
set(ORT_DIR "${CMAKE_SOURCE_DIR}/third-party")

# x64 Linux/Windows default to the CUDA build; pass -DCPP_ONNX_PYTHON_ORT_VARIANT=cpu
# to ship the much smaller CPU-only runtime instead.
set(CPP_ONNX_PYTHON_ORT_VARIANT "gpu" CACHE STRING
    "ONNX Runtime build variant for x64: gpu (CUDA) or cpu")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    set(ORT_ARCH "arm64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    set(ORT_ARCH "x64")
else()
    message(FATAL_ERROR "Unsupported processor '${CMAKE_SYSTEM_PROCESSOR}'")
endif()

# Per-OS/arch directory, so building a Linux wheel in a container does not
# clobber the host's macOS extraction (and vice versa).
set(ORT_FINAL_DIR "${ORT_DIR}/onnxruntime-${CMAKE_SYSTEM_NAME}-${ORT_ARCH}")

if(WIN32)
    if(ORT_ARCH STREQUAL "arm64")
        set(ORT_ARCHIVE "onnxruntime-win-arm64x-${ORT_VERSION}.zip")
    elseif(CPP_ONNX_PYTHON_ORT_VARIANT STREQUAL "cpu")
        set(ORT_ARCHIVE "onnxruntime-win-x64-${ORT_VERSION}.zip")
    else()
        set(ORT_ARCHIVE "onnxruntime-win-x64-gpu_cuda12-${ORT_VERSION}.zip")
    endif()
    set(ORT_LIB_NAME "onnxruntime")
    set(ORT_LIBRARY_FILENAME "onnxruntime.lib")
    set(ORT_RUNTIME_DLL "${ORT_FINAL_DIR}/lib/onnxruntime.dll")
elseif(APPLE)
    if(NOT ORT_ARCH STREQUAL "arm64")
        message(FATAL_ERROR
            "ONNX Runtime ${ORT_VERSION} only publishes osx-arm64 macOS builds")
    endif()
    set(ORT_ARCHIVE "onnxruntime-osx-arm64-${ORT_VERSION}.tgz")
    set(ORT_LIB_NAME "libonnxruntime")
    set(ORT_LIBRARY_FILENAME "libonnxruntime.dylib")
elseif(UNIX)
    if(ORT_ARCH STREQUAL "arm64")
        set(ORT_ARCHIVE "onnxruntime-linux-aarch64-${ORT_VERSION}.tgz")
    elseif(CPP_ONNX_PYTHON_ORT_VARIANT STREQUAL "cpu")
        set(ORT_ARCHIVE "onnxruntime-linux-x64-${ORT_VERSION}.tgz")
    else()
        set(ORT_ARCHIVE "onnxruntime-linux-x64-gpu_cuda12-${ORT_VERSION}.tgz")
    endif()
    set(ORT_LIB_NAME "libonnxruntime")
    set(ORT_LIBRARY_FILENAME "libonnxruntime.so")
else()
    message(FATAL_ERROR "Unsupported platform for ONNX Runtime bootstrap")
endif()

set(ORT_LIB_FILE "${ORT_FINAL_DIR}/lib/${ORT_LIBRARY_FILENAME}")
set(ORT_IMPLIB "${ORT_FINAL_DIR}/lib/onnxruntime.lib")

# Optional integrity pin; the macOS arm64 archive is pinned by default.
set(CPP_ONNX_PYTHON_ORT_SHA256 "" CACHE STRING
    "Expected SHA256 of the ONNX Runtime archive (empty disables the check)")
if(NOT CPP_ONNX_PYTHON_ORT_SHA256 AND APPLE AND ORT_ARCH STREQUAL "arm64")
    set(CPP_ONNX_PYTHON_ORT_SHA256
        "6ebb5062a934537c352937821f9fe9718e7de1a2db1122a93dd363ffd53a7012")
endif()
set(ORT_HASH_ARGS "")
if(CPP_ONNX_PYTHON_ORT_SHA256)
    set(ORT_HASH_ARGS EXPECTED_HASH SHA256=${CPP_ONNX_PYTHON_ORT_SHA256})
endif()

if(EXISTS "${ORT_FINAL_DIR}/include/onnxruntime_c_api.h" AND EXISTS "${ORT_LIB_FILE}")
    message(STATUS "ONNX Runtime ${ORT_VERSION} already bootstrapped; reusing")
else()
    set(ORT_URL
        "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_ARCHIVE}")
    set(ORT_ARCHIVE_PATH "${ORT_DIR}/${ORT_ARCHIVE}")
    set(ORT_EXTRACT_DIR "${CMAKE_BINARY_DIR}/ort-extract-${ORT_VERSION}")
    message(STATUS "Bootstrapping ONNX Runtime ${ORT_VERSION}")

    # The archive lives in the source tree so it survives scikit-build-core's
    # per-wheel build directories and is downloaded only once.
    if(EXISTS "${ORT_ARCHIVE_PATH}")
        message(STATUS "Reusing downloaded archive ${ORT_ARCHIVE_PATH}")
    else()
        if(NOT ORT_HASH_ARGS)
            message(WARNING
                "No SHA256 pinned for ${ORT_ARCHIVE}; the download is not hash-verified")
        endif()
        file(DOWNLOAD "${ORT_URL}" "${ORT_ARCHIVE_PATH}"
            SHOW_PROGRESS TLS_VERIFY ON ${ORT_HASH_ARGS} STATUS ORT_DOWNLOAD_STATUS)
        list(GET ORT_DOWNLOAD_STATUS 0 ORT_DOWNLOAD_CODE)
        list(GET ORT_DOWNLOAD_STATUS 1 ORT_DOWNLOAD_MESSAGE)
        if(NOT ORT_DOWNLOAD_CODE EQUAL 0)
            file(REMOVE "${ORT_ARCHIVE_PATH}")
            message(FATAL_ERROR
                "Failed to download ${ORT_URL}: ${ORT_DOWNLOAD_MESSAGE}")
        endif()
    endif()

    # Extract into scratch, then move the single result into place.
    file(REMOVE_RECURSE "${ORT_EXTRACT_DIR}")
    file(ARCHIVE_EXTRACT INPUT "${ORT_ARCHIVE_PATH}" DESTINATION "${ORT_EXTRACT_DIR}")
    file(GLOB ORT_EXTRACTED LIST_DIRECTORIES true "${ORT_EXTRACT_DIR}/onnxruntime-*")
    list(LENGTH ORT_EXTRACTED ORT_EXTRACTED_COUNT)
    if(NOT ORT_EXTRACTED_COUNT EQUAL 1)
        file(REMOVE_RECURSE "${ORT_EXTRACT_DIR}")
        message(FATAL_ERROR
            "Expected one extracted directory, found ${ORT_EXTRACTED_COUNT}")
    endif()
    file(REMOVE_RECURSE "${ORT_FINAL_DIR}")
    file(RENAME "${ORT_EXTRACTED}" "${ORT_FINAL_DIR}")
    file(REMOVE_RECURSE "${ORT_EXTRACT_DIR}")

    if(NOT EXISTS "${ORT_LIB_FILE}")
        message(FATAL_ERROR "Extraction finished but '${ORT_LIB_FILE}' is missing")
    endif()
endif()

add_library(onnxruntime SHARED IMPORTED GLOBAL)
if(WIN32)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_IMPLIB "${ORT_IMPLIB}" IMPORTED_LOCATION "${ORT_RUNTIME_DLL}")
else()
    # The linker reads the real SONAME / install name from the library.
    set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION "${ORT_LIB_FILE}")
endif()
set_target_properties(onnxruntime PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_FINAL_DIR}/include")
