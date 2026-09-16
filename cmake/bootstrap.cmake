cmake_minimum_required(VERSION 3.20)

set(ORT_VERSION "1.30.0")
set(DEPS_DIR ${CMAKE_SOURCE_DIR}/third-party/deps/)
set(ORT_DIR "${DEPS_DIR}/onnxruntime")

message(STATUS "looking for ${ORT_DIR}/include/onnxruntime_cxx_api.h")
if(EXISTS "${ORT_DIR}/include/onnxruntime_cxx_api.h")
    message(STATUS "ONNX Runtime already installed")
    return()
endif()

if(WIN32)
    set(ORT_PLATFORM "win")
elseif(APPLE)
    set(ORT_PLATFORM "osx")
elseif(UNIX)
    set(ORT_PLATFORM "linux")
else()
    message(FATAL_ERROR "Unsupported platform")
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    set(ORT_ARCH "arm64")
else()
    set(ORT_ARCH "x64")
endif()

if(WIN32)
    set(ARCHIVE_EXT ".zip")
else()
    set(ARCHIVE_EXT ".tgz")
endif()
set(ARCHIVE "onnxruntime-${ORT_PLATFORM}-${ORT_ARCH}-${ORT_VERSION}${ARCHIVE_EXT}")
set(URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ARCHIVE}")
set(ARCHIVE_PATH "${DEPS_DIR}/${ARCHIVE}")
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
set(ORT_EXTRACTED_DIR "${DEPS_DIR}/onnxruntime-${ORT_PLATFORM}-${ORT_ARCH}-${ORT_VERSION}")
file(RENAME
    "${ORT_EXTRACTED_DIR}"
    "${ORT_DIR}"
)

file(REMOVE "${ARCHIVE_PATH}")

message(STATUS "ONNX Runtime installed in ${ORT_DIR}")
