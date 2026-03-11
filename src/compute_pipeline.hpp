#pragma once

#include "buffer_utils.hpp"
#include "shader_utils.hpp"
#include "vulkan_context.hpp"

#include <cstdint>
#include <filesystem>
#include <vector>

struct DispatchConfig {
  uint32_t workgroups = 0;
  uint32_t rounds = 0;
};

class ComputePipeline {
 public:
  ComputePipeline(const VulkanContext& context,
                  const std::filesystem::path& shader_path,
                  uint32_t output_words);
  ~ComputePipeline();

  ComputePipeline(const ComputePipeline&) = delete;
  ComputePipeline& operator=(const ComputePipeline&) = delete;

  std::vector<uint32_t> Run(const DispatchConfig& config, bool verbose);
  const SpirvInstructionSummary& spirv_summary() const { return spirv_summary_; }
  const std::filesystem::path& shader_path() const { return shader_path_; }

 private:
  struct PushConstants {
    uint32_t rounds = 0;
  };

  const VulkanContext& context_;
  std::filesystem::path shader_path_;
  SpirvInstructionSummary spirv_summary_;

  BufferAllocation output_buffer_;
  VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
  VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
  VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
};
