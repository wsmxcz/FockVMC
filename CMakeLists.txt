cmake_minimum_required(VERSION 3.20)

project(detnqs LANGUAGES CXX)

include(FetchContent)

option(DETNQS_BUILD_LIBDET "Build libdet Python extension" ON)

FetchContent_Declare(
  unordered_dense
  GIT_REPOSITORY https://github.com/martinus/unordered_dense.git
  GIT_TAG v4.8.1
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(unordered_dense)

if(DETNQS_BUILD_LIBDET)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(nanobind CONFIG REQUIRED)

  nanobind_add_module(_libdet_cpp MODULE
    libdet/csrc/connections.cpp
    libdet/csrc/external.cpp
    libdet/csrc/hamiltonian.cpp
    libdet/python/module.cpp
  )

  target_include_directories(_libdet_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/libdet/csrc
  )

  target_compile_features(_libdet_cpp PRIVATE cxx_std_20)

  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(_libdet_cpp PRIVATE OpenMP::OpenMP_CXX)
  endif()

  if(TARGET unordered_dense::unordered_dense)
    target_link_libraries(_libdet_cpp PRIVATE unordered_dense::unordered_dense)
  elseif(TARGET unordered_dense)
    target_link_libraries(_libdet_cpp PRIVATE unordered_dense)
  endif()

  install(TARGETS _libdet_cpp
    LIBRARY DESTINATION libdet
    RUNTIME DESTINATION libdet
  )
endif()
