cmake_minimum_required(VERSION 3.24)
project(DspSandbox VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 20)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_executable(audio-engine dsp/sampler.cpp dsp/convolution_mt.cpp dsp/fft.cpp)

# Emscripten linker flags
target_link_options(
  audio-engine
  PRIVATE
  --bind
  "SHELL:-s MODULARIZE=1"
  "SHELL:-s EXPORT_NAME=createAudioEngine"
  "SHELL:-s ENVIRONMENT=web,worker,shell"
  "SHELL:-s SINGLE_FILE=1"
  "SHELL:-s EXPORTED_FUNCTIONS=['_malloc','_free']"
  "SHELL:-s EXPORTED_RUNTIME_METHODS=['ccall','cwrap','HEAPF32']"
  "SHELL:-s ALLOW_MEMORY_GROWTH")

# Output into frontend/public/
set_target_properties(
  audio-engine
  PROPERTIES OUTPUT_NAME "audio-engine"
             SUFFIX ".js"
             RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/frontend/public")

# Tail engine — runs in a Web Worker, handles convolution levels 1..N
add_executable(tail-engine dsp/tail_worker.cpp dsp/convolution_mt.cpp dsp/fft.cpp)

target_link_options(
  tail-engine
  PRIVATE
  --bind
  "SHELL:-s MODULARIZE=1"
  "SHELL:-s EXPORT_NAME=createTailEngine"
  "SHELL:-s ENVIRONMENT=worker"
  "SHELL:-s SINGLE_FILE=1"
  "SHELL:-s EXPORTED_FUNCTIONS=['_malloc','_free']"
  "SHELL:-s EXPORTED_RUNTIME_METHODS=['ccall','cwrap','HEAPF32']"
  "SHELL:-s ALLOW_MEMORY_GROWTH")

set_target_properties(
  tail-engine
  PROPERTIES OUTPUT_NAME "tail-engine"
             SUFFIX ".js"
             RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/frontend/public")
