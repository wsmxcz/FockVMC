cmake_minimum_required(VERSION 3.20)

project(detnqs LANGUAGES CXX)

include(FetchContent)

FetchContent_Declare(
  unordered_dense
  GIT_REPOSITORY https://github.com/martinus/unordered_dense.git
  GIT_TAG v4.8.1
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(unordered_dense)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(libdet MODULE
  detnqs/operator/libdet/module.cpp
)

target_include_directories(libdet PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/detnqs/operator/libdet/csrc
)

target_compile_features(libdet PRIVATE cxx_std_20)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(libdet PRIVATE OpenMP::OpenMP_CXX)
endif()

if(TARGET unordered_dense::unordered_dense)
  target_link_libraries(libdet PRIVATE unordered_dense::unordered_dense)
elseif(TARGET unordered_dense)
  target_link_libraries(libdet PRIVATE unordered_dense)
endif()

install(TARGETS libdet
  LIBRARY DESTINATION detnqs/operator
  RUNTIME DESTINATION detnqs/operator
)
