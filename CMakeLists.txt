cmake_minimum_required(VERSION 3.20)

project(detnqs LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

option(DETNQS_BUILD_PYTHON "Build the nanobind Python extension" ON)
option(DETNQS_BUILD_CPP_TESTS "Build C++ libdet tests" OFF)

set(
  DETNQS_ANKERL_GIT_TAG
  "v4.8.1"
  CACHE STRING
  "ankerl::unordered_dense git tag used when FetchContent is needed"
)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

find_package(OpenMP REQUIRED)
find_package(unordered_dense CONFIG QUIET)

if(NOT TARGET unordered_dense::unordered_dense)
  include(FetchContent)

  FetchContent_Declare(
    unordered_dense
    GIT_REPOSITORY https://github.com/martinus/unordered_dense.git
    GIT_TAG ${DETNQS_ANKERL_GIT_TAG}
    GIT_SHALLOW TRUE
  )

  FetchContent_MakeAvailable(unordered_dense)
endif()

if(NOT TARGET unordered_dense::unordered_dense)
  message(FATAL_ERROR "unordered_dense::unordered_dense target is required")
endif()

# ---------------------------------------------------------------------------
# libdet headers and common compile settings
# ---------------------------------------------------------------------------

set(DETNQS_LIBDET_CSRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/detnqs/operator/libdet/csrc)
set(DETNQS_LIBDET_MODULE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/detnqs/operator/libdet/module.cpp)
set(DETNQS_PYTHON_EXTENSION_INSTALL_DIR detnqs/operator)

add_library(libdet_common INTERFACE)

target_compile_features(libdet_common INTERFACE cxx_std_20)
target_include_directories(libdet_common INTERFACE
  ${DETNQS_LIBDET_CSRC_DIR}
)
target_link_libraries(libdet_common INTERFACE
  OpenMP::OpenMP_CXX
  unordered_dense::unordered_dense
)

# ---------------------------------------------------------------------------
# Python extension
# ---------------------------------------------------------------------------

if(DETNQS_BUILD_PYTHON)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(nanobind CONFIG REQUIRED)

  nanobind_add_module(libdet MODULE ${DETNQS_LIBDET_MODULE_SRC})
  target_link_libraries(libdet PRIVATE libdet_common)

  install(
    TARGETS libdet
    LIBRARY DESTINATION ${DETNQS_PYTHON_EXTENSION_INSTALL_DIR}
    RUNTIME DESTINATION ${DETNQS_PYTHON_EXTENSION_INSTALL_DIR}
  )
endif()

# ---------------------------------------------------------------------------
# C++ tests
# ---------------------------------------------------------------------------

if(DETNQS_BUILD_CPP_TESTS)
  enable_testing()

  add_executable(test_rhf tests/cpp/test_rhf.cpp)
  target_link_libraries(test_rhf PRIVATE libdet_common)
  add_test(NAME test_rhf COMMAND test_rhf)
endif()
