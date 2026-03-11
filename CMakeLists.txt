cmake_minimum_required(VERSION 3.20)

project(moltenvk_barrier_repro LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(cmake/DetectVulkanSDK.cmake)

set(REPRO_SHADER_SOURCES
    barrier_only.comp
    memorybarrier_plus_barrier.comp
    groupmemorybarrier_plus_barrier.comp)

set(REPRO_SHADER_OUTPUTS)
foreach(shader_file IN LISTS REPRO_SHADER_SOURCES)
  get_filename_component(shader_name "${shader_file}" NAME)
  set(shader_output "${CMAKE_BINARY_DIR}/shaders/${shader_name}.spv")
  add_custom_command(
      OUTPUT "${shader_output}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_BINARY_DIR}/shaders"
      COMMAND "${REPRO_GLSLANG_VALIDATOR}"
              -V
              --target-env
              vulkan1.1
              "-I${CMAKE_SOURCE_DIR}/shaders"
              -o
              "${shader_output}"
              "${CMAKE_SOURCE_DIR}/shaders/${shader_file}"
      DEPENDS
          "${CMAKE_SOURCE_DIR}/shaders/${shader_file}"
          "${CMAKE_SOURCE_DIR}/shaders/common.glsl"
      VERBATIM)
  list(APPEND REPRO_SHADER_OUTPUTS "${shader_output}")
endforeach()

add_custom_target(repro_shaders ALL DEPENDS ${REPRO_SHADER_OUTPUTS})

add_executable(moltenvk_barrier_repro
    src/main.cpp
    src/vulkan_context.cpp
    src/compute_pipeline.cpp
    src/buffer_utils.cpp
    src/shader_utils.cpp)

target_include_directories(moltenvk_barrier_repro PRIVATE src)
target_link_libraries(moltenvk_barrier_repro PRIVATE Vulkan::Vulkan)
target_compile_definitions(moltenvk_barrier_repro PRIVATE
    REPRO_DEFAULT_SHADER_DIR="${CMAKE_BINARY_DIR}/shaders")
add_dependencies(moltenvk_barrier_repro repro_shaders)

add_custom_target(repro_run
    COMMAND "${CMAKE_SOURCE_DIR}/scripts/run_repro.sh" "${CMAKE_BINARY_DIR}"
    DEPENDS moltenvk_barrier_repro
    USES_TERMINAL)
