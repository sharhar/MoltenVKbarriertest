#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

std::vector<uint32_t> ReadSpirvFile(const std::filesystem::path& path);

VkShaderModule CreateShaderModule(VkDevice device, const std::vector<uint32_t>& spirv_words);

struct SpirvInstructionSummary {
  size_t word_count = 0;
  size_t instruction_count = 0;
  size_t control_barrier_count = 0;
  size_t memory_barrier_count = 0;
};

SpirvInstructionSummary SummarizeSpirv(const std::vector<uint32_t>& spirv_words);

std::string FormatSpirvSummary(const SpirvInstructionSummary& summary);
