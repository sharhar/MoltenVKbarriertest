include_guard(GLOBAL)

find_package(Vulkan REQUIRED)

find_program(REPRO_GLSLANG_VALIDATOR
    NAMES glslangValidator
    HINTS
      "$ENV{VULKAN_SDK}/bin"
      "$ENV{VULKAN_SDK}/Bin")

if(NOT REPRO_GLSLANG_VALIDATOR)
  message(FATAL_ERROR "glslangValidator was not found. Install the LunarG Vulkan SDK or add glslangValidator to PATH.")
endif()

find_program(REPRO_SPIRV_DIS
    NAMES spirv-dis
    HINTS
      "$ENV{VULKAN_SDK}/bin"
      "$ENV{VULKAN_SDK}/Bin")

if(REPRO_SPIRV_DIS)
  message(STATUS "Found spirv-dis: ${REPRO_SPIRV_DIS}")
else()
  message(STATUS "spirv-dis not found; scripts/dump_spirv.sh will require it on PATH.")
endif()
