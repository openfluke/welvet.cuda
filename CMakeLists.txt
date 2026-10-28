cmake_minimum_required(VERSION 3.20)

# Normal C++ host + nvcc CUDA (same shape as a standard CUDA library).
# Default arch: Turing (sm_75). Override: -DCMAKE_CUDA_ARCHITECTURES=86;89
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES "75")
endif()

project(welvet_cuda VERSION 0.56 LANGUAGES CXX CUDA)

# NVIDIA CUDA-only Welvet engine: C++ host + CUDA kernels.
# Linux x86_64 is the primary host; Windows is a later artifact.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(WELVET_BUILD_SHARED "Build shared library (C ABI / NAC)" ON)
option(WELVET_BUILD_TOOLS "Build CLI tools for w2a go_v_cuda" ON)
option(WELVET_BUILD_EXAMPLES "Build standalone CUDA smoke examples" ON)

find_package(CUDAToolkit REQUIRED)

include(GNUInstallDirs)

set(WELVET_CORE_SOURCES
  src/core/status.cpp
  src/core/dtype.cpp
  src/core/activate.cpp
  src/core/tensor.cpp
  src/core/float_codecs.cpp
  src/cuda/device.cpp
  src/cuda/buffer.cpp
  src/cuda/kernels.cu
  src/weights/store.cpp
  src/weights/native_codec.cpp
  src/layers/dense/dense.cpp
  src/layers/dense/session.cpp
  src/layers/rmsnorm/rmsnorm.cpp
  src/layers/layernorm/layernorm.cpp
  src/layers/softmax/softmax.cpp
  src/layers/swiglu/swiglu.cpp
  src/layers/embedding/embedding.cpp
  src/layers/mha/mha.cpp
  src/layers/cnn1/cnn1.cpp
  src/layers/cnn2/cnn2.cpp
  src/layers/cnn3/cnn3.cpp
  src/layers/convt1/convt1.cpp
  src/layers/convt2/convt2.cpp
  src/layers/convt3/convt3.cpp
  src/layers/rnn/rnn.cpp
  src/layers/lstm/lstm.cpp
  src/layers/sequential/sequential.cpp
  src/layers/residual/residual.cpp
  src/layers/stack/stack.cpp
  src/layers/mamba/mamba.cpp
  src/layers/metacognition/metacognition.cpp
  src/layers/gdn/gdn.cpp
  src/layers/kmeans/kmeans.cpp
  src/layers/parallel/parallel.cpp
  src/architecture/grid.cpp
  src/runtime/forward.cpp
  src/runtime/backward.cpp
  src/runtime/training.cpp
  src/runtime/line.cpp
  src/runtime/step.cpp
  src/runtime/tween_train.cpp
  src/runtime/tween_chain_train.cpp
  src/runtime/tween_split_train.cpp
  src/systems/tween.cpp
  src/systems/tween_chain.cpp
  src/systems/tween_split.cpp
  src/gaps/registry.cpp
  src/interop/c_abi.cpp
)

add_library(welvet_cuda_obj OBJECT ${WELVET_CORE_SOURCES})
set_target_properties(welvet_cuda_obj PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(welvet_cuda_obj
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_definitions(welvet_cuda_obj PRIVATE WELVET_CUDA=1)
target_link_libraries(welvet_cuda_obj PUBLIC CUDA::cudart)
# Match welvet.cpp / Go: disable FMA contraction so host Softmax/KMeans/Parallel
# double paths stay bit-stable vs Go (see kmeans.cpp comment).
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(welvet_cuda_obj PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
endif()

if(WELVET_BUILD_SHARED)
  add_library(welvet SHARED $<TARGET_OBJECTS:welvet_cuda_obj>)
else()
  add_library(welvet STATIC $<TARGET_OBJECTS:welvet_cuda_obj>)
endif()
set_target_properties(welvet PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_library(welvet::welvet ALIAS welvet)
target_include_directories(welvet
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(welvet PUBLIC CUDA::cudart)
set_target_properties(welvet PROPERTIES
  OUTPUT_NAME welvet_cuda
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ON
)

if(WELVET_BUILD_TOOLS)
  add_executable(welvet_dense_fwd
    tools/dense_fwd/main.cpp
    tools/dense_fwd/volume_fwd.cpp
    tools/dense_fwd/stepbp_wires.cpp
  )
  target_include_directories(welvet_dense_fwd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tools/dense_fwd)
  target_link_libraries(welvet_dense_fwd PRIVATE welvet)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(welvet_dense_fwd PRIVATE -ffp-contract=off)
  endif()
endif()

if(WELVET_BUILD_EXAMPLES)
  add_executable(example_cuda_smoke examples/cuda_smoke/main.cpp)
  target_link_libraries(example_cuda_smoke PRIVATE welvet)
endif()

install(TARGETS welvet
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY kernels/ DESTINATION ${CMAKE_INSTALL_DATADIR}/welvet/kernels)
if(WELVET_BUILD_TOOLS)
  install(TARGETS welvet_dense_fwd RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

message(STATUS "welvet.cuda platform: ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CUDAToolkit: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA archs: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CXX: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
