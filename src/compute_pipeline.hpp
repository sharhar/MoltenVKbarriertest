#pragma once

#include "buffer_utils.hpp"
#include "shader_utils.hpp"
#include "vulkan_context.hpp"

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

struct ComplexValue {
  float real = 0.0f;
  float imag = 0.0f;
};

struct DispatchConfig {
  uint32_t workgroups = 0;
};

class ComputePipeline {
 public:
  ComputePipeline(const VulkanContext& context,
                  const std::filesystem::path& shader_path,
                  uint32_t complex_value_count);
  ~ComputePipeline();

  ComputePipeline(const ComputePipeline&) = delete;
  ComputePipeline& operator=(const ComputePipeline&) = delete;

  std::vector<ComplexValue> Run(const DispatchConfig& config,
                                std::span<const ComplexValue> input_data);
  const SpirvInstructionSummary& spirv_summary() const { return spirv_summary_; }
  const std::filesystem::path& shader_path() const { return shader_path_; }

 private:
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
