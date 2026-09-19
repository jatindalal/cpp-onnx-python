cmake_minimum_required(VERSION 3.15)

set(ORT_VERSION "1.30.0")
set(DEPS_DIR ${CMAKE_SOURCE_DIR}/third-party/deps/)
set(ORT_DIR "${DEPS_DIR}/onnxruntime")

if(WIN32)
    set(ORT_PLATFORM "win")
    set(ARCHIVE_EXT ".zip")
elseif(APPLE)
    set(ORT_PLATFORM "osx")
    set(ARCHIVE_EXT ".tgz")
elseif(UNIX)
    set(ORT_PLATFORM "linux")
    set(ARCHIVE_EXT ".tgz")
else()
    message(FATAL_ERROR "Unsupported platform")
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    set(ORT_ARCH "arm64")
else()
    set(ORT_ARCH "x64")
endif()

set(ARCHIVE "onnxruntime-${ORT_PLATFORM}-${ORT_ARCH}-${ORT_VERSION}${ARCHIVE_EXT}")
set(URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ARCHIVE}")
set(ARCHIVE_PATH "${DEPS_DIR}/${ARCHIVE}")
set(ORT_EXTRACTED_DIR "${DEPS_DIR}/onnxruntime-${ORT_PLATFORM}-${ORT_ARCH}-${ORT_VERSION}")

# Platform-specific library that the imported target links against.
if(WIN32)
    set(ORT_IMPLIB "${ORT_DIR}/lib/onnxruntime.lib")
    set(ORT_RUNTIME_DLL "${ORT_DIR}/lib/onnxruntime.dll")
    set(ORT_LIB_FILE "${ORT_IMPLIB}")
else()
    set(ORT_LIB_NAME "libonnxruntime")
    set(ORT_LIB_FILE "${ORT_DIR}/lib/${ORT_LIB_NAME}.dylib")
    if(NOT APPLE)
        set(ORT_LIB_FILE "${ORT_DIR}/lib/${ORT_LIB_NAME}.so")
    endif()
endif()

message(STATUS "looking for ${ORT_LIB_FILE}")
if(EXISTS "${ORT_LIB_FILE}")
    message(STATUS "ONNX Runtime already installed")
    return()
endif()

message(STATUS "Downloading ONNX Runtime ${ORT_VERSION}")
message(STATUS "  ${URL}")

file(DOWNLOAD
    "${URL}"
    "${ARCHIVE_PATH}"
    SHOW_PROGRESS
    STATUS DOWNLOAD_STATUS
    TLS_VERIFY ON
)
list(GET DOWNLOAD_STATUS 0 DOWNLOAD_CODE)
if(NOT DOWNLOAD_CODE EQUAL 0)
    message(FATAL_ERROR
        "Failed to download ONNX Runtime: ${DOWNLOAD_STATUS}"
    )
endif()

if(WIN32)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E unzip "${ARCHIVE_PATH}"
        WORKING_DIRECTORY "${DEPS_DIR}"
        RESULT_VARIABLE RESULT
    )
else()
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E tar xzf "${ARCHIVE_PATH}"
        WORKING_DIRECTORY "${DEPS_DIR}"
        RESULT_VARIABLE RESULT
    )
endif()

if(NOT RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to extract ONNX Runtime")
endif()

file(RENAME
    "${ORT_EXTRACTED_DIR}"
    "${ORT_DIR}"
)

file(REMOVE "${ARCHIVE_PATH}")

if(NOT EXISTS "${ORT_LIB_FILE}")
    message(FATAL_ERROR
        "Extracted ONNX Runtime does not contain the expected library: ${ORT_LIB_FILE}"
    )
endif()

message(STATUS "ONNX Runtime installed in ${ORT_DIR}")
