cmake_minimum_required(VERSION 3.20)

project(detnqs LANGUAGES CXX)

option(DETNQS_BUILD_PYTHON "Build the nanobind Python extension" ON)
option(DETNQS_BUILD_CPP_TESTS "Build C++ libdet tests" OFF)

find_package(OpenMP REQUIRED)
find_package(unordered_dense CONFIG QUIET)

if(NOT TARGET unordered_dense::unordered_dense)
  include(FetchContent)

  FetchContent_Declare(
    unordered_dense
    GIT_REPOSITORY https://github.com/martinus/unordered_dense.git
    GIT_TAG v4.8.1
    GIT_SHALLOW TRUE
  )

  FetchContent_MakeAvailable(unordered_dense)
endif()

add_library(libdet_common INTERFACE)

target_compile_features(libdet_common INTERFACE cxx_std_20)
target_include_directories(libdet_common INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/detnqs/operator/libdet/csrc
)
target_link_libraries(libdet_common INTERFACE
  OpenMP::OpenMP_CXX
  unordered_dense::unordered_dense
)

if(DETNQS_BUILD_PYTHON)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(nanobind CONFIG REQUIRED)

  nanobind_add_module(
    libdet
    MODULE
    ${CMAKE_CURRENT_SOURCE_DIR}/detnqs/operator/libdet/module.cpp
  )
  target_link_libraries(libdet PRIVATE libdet_common)

  install(
    TARGETS libdet
    LIBRARY DESTINATION detnqs/operator
    RUNTIME DESTINATION detnqs/operator
  )
endif()

if(DETNQS_BUILD_CPP_TESTS)
  enable_testing()

  add_executable(test_rhf tests/cpp/test_rhf.cpp)
  target_link_libraries(test_rhf PRIVATE libdet_common)
  add_test(NAME test_rhf COMMAND test_rhf)
endif()
