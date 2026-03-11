#include "shader_utils.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {
constexpr uint16_t kOpControlBarrier = 224;
constexpr uint16_t kOpMemoryBarrier = 225;
}  // namespace

std::vector<uint32_t> ReadSpirvFile(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("Failed to open SPIR-V file: " + path.string());
  }

  const std::streamsize size_bytes = file.tellg();
  if (size_bytes <= 0 || (size_bytes % static_cast<std::streamsize>(sizeof(uint32_t))) != 0) {
    throw std::runtime_error("SPIR-V file has an invalid size: " + path.string());
  }
  file.seekg(0, std::ios::beg);

  std::vector<uint32_t> words(static_cast<size_t>(size_bytes) / sizeof(uint32_t));
  if (!file.read(reinterpret_cast<char*>(words.data()), size_bytes)) {
    throw std::runtime_error("Failed to read SPIR-V file: " + path.string());
  }

  return words;
}

VkShaderModule CreateShaderModule(VkDevice device, const std::vector<uint32_t>& spirv_words) {
  VkShaderModuleCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  create_info.codeSize = spirv_words.size() * sizeof(uint32_t);
  create_info.pCode = spirv_words.data();

  VkShaderModule shader_module = VK_NULL_HANDLE;
  if (vkCreateShaderModule(device, &create_info, nullptr, &shader_module) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create shader module.");
  }

  return shader_module;
}

SpirvInstructionSummary SummarizeSpirv(const std::vector<uint32_t>& spirv_words) {
  SpirvInstructionSummary summary{};
  summary.word_count = spirv_words.size();

  if (spirv_words.size() < 5) {
    return summary;
  }

  size_t index = 5;
  while (index < spirv_words.size()) {
    const uint32_t instruction_header = spirv_words[index];
    const uint16_t word_count = static_cast<uint16_t>(instruction_header >> 16);
    const uint16_t opcode = static_cast<uint16_t>(instruction_header & 0xffffu);

    if (word_count == 0) {
      break;
    }

    ++summary.instruction_count;
    if (opcode == kOpControlBarrier) {
      ++summary.control_barrier_count;
    } else if (opcode == kOpMemoryBarrier) {
      ++summary.memory_barrier_count;
    }
    index += word_count;
  }

  return summary;
}

std::string FormatSpirvSummary(const SpirvInstructionSummary& summary) {
  std::ostringstream stream;
  stream << "SPIR-V words: " << summary.word_count << "\n";
  stream << "SPIR-V instructions: " << summary.instruction_count << "\n";
  stream << "OpControlBarrier count: " << summary.control_barrier_count << "\n";
  stream << "OpMemoryBarrier count: " << summary.memory_barrier_count;
  return stream.str();
}
