#include "compute_pipeline.hpp"

#include <cstring>
#include <stdexcept>

ComputePipeline::ComputePipeline(const VulkanContext& context,
                                 const std::filesystem::path& shader_path,
                                 uint32_t complex_value_count)
    : context_(context), shader_path_(shader_path) {
  output_buffer_ = CreateBuffer(
      context_.physical_device(),
      context_.device(),
      static_cast<VkDeviceSize>(complex_value_count) * sizeof(ComplexValue),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  VkDescriptorSetLayoutBinding binding{};
  binding.binding = 0;
  binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  binding.descriptorCount = 1;
  binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info.bindingCount = 1;
  layout_info.pBindings = &binding;

  if (vkCreateDescriptorSetLayout(context_.device(), &layout_info, nullptr,
                                  &descriptor_set_layout_) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create descriptor set layout.");
  }

  VkDescriptorPoolSize pool_size{};
  pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  pool_size.descriptorCount = 1;

  VkDescriptorPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.maxSets = 1;
  pool_info.poolSizeCount = 1;
  pool_info.pPoolSizes = &pool_size;

  if (vkCreateDescriptorPool(context_.device(), &pool_info, nullptr, &descriptor_pool_) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create descriptor pool.");
  }

  VkDescriptorSetAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  alloc_info.descriptorPool = descriptor_pool_;
  alloc_info.descriptorSetCount = 1;
  alloc_info.pSetLayouts = &descriptor_set_layout_;

  if (vkAllocateDescriptorSets(context_.device(), &alloc_info, &descriptor_set_) != VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate descriptor set.");
  }

  VkDescriptorBufferInfo buffer_info{};
  buffer_info.buffer = output_buffer_.buffer;
  buffer_info.offset = 0;
  buffer_info.range = output_buffer_.size;

  VkWriteDescriptorSet write{};
  write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.dstSet = descriptor_set_;
  write.dstBinding = 0;
  write.descriptorCount = 1;
  write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  write.pBufferInfo = &buffer_info;
  vkUpdateDescriptorSets(context_.device(), 1, &write, 0, nullptr);

  VkPipelineLayoutCreateInfo pipeline_layout_info{};
  pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_info.setLayoutCount = 1;
  pipeline_layout_info.pSetLayouts = &descriptor_set_layout_;

  if (vkCreatePipelineLayout(context_.device(), &pipeline_layout_info, nullptr,
                             &pipeline_layout_) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create pipeline layout.");
  }

  const std::vector<uint32_t> spirv_words = ReadSpirvFile(shader_path_);
  spirv_summary_ = SummarizeSpirv(spirv_words);
  const VkShaderModule shader_module = CreateShaderModule(context_.device(), spirv_words);

  VkPipelineShaderStageCreateInfo shader_stage{};
  shader_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shader_stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  shader_stage.module = shader_module;
  shader_stage.pName = "main";

  VkComputePipelineCreateInfo pipeline_info{};
  pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeline_info.stage = shader_stage;
  pipeline_info.layout = pipeline_layout_;

  if (vkCreateComputePipelines(context_.device(), VK_NULL_HANDLE, 1, &pipeline_info, nullptr,
                               &pipeline_) != VK_SUCCESS) {
    vkDestroyShaderModule(context_.device(), shader_module, nullptr);
    throw std::runtime_error("Failed to create compute pipeline.");
  }

  vkDestroyShaderModule(context_.device(), shader_module, nullptr);
}

ComputePipeline::~ComputePipeline() {
  if (pipeline_ != VK_NULL_HANDLE) {
    vkDestroyPipeline(context_.device(), pipeline_, nullptr);
  }
  if (pipeline_layout_ != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(context_.device(), pipeline_layout_, nullptr);
  }
  if (descriptor_pool_ != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(context_.device(), descriptor_pool_, nullptr);
  }
  if (descriptor_set_layout_ != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(context_.device(), descriptor_set_layout_, nullptr);
  }
  DestroyBuffer(context_.device(), output_buffer_);
}

std::vector<ComplexValue> ComputePipeline::Run(const DispatchConfig& config,
                                               std::span<const ComplexValue> input_data) {
  if (input_data.size_bytes() != output_buffer_.size) {
    throw std::runtime_error("Input blob size does not match the Vulkan storage buffer size.");
  }

  std::memcpy(output_buffer_.mapped, input_data.data(), input_data.size_bytes());

  VkCommandBufferAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.commandPool = context_.command_pool();
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandBufferCount = 1;

  VkCommandBuffer command_buffer = VK_NULL_HANDLE;
  if (vkAllocateCommandBuffers(context_.device(), &alloc_info, &command_buffer) != VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate command buffer.");
  }

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  if (vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS) {
    vkFreeCommandBuffers(context_.device(), context_.command_pool(), 1, &command_buffer);
    throw std::runtime_error("Failed to begin command buffer.");
  }

  vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
  vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1,
                          &descriptor_set_, 0, nullptr);
  vkCmdDispatch(command_buffer, config.workgroups, 1, 1);

  VkBufferMemoryBarrier host_barrier{};
  host_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  host_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  host_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  host_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  host_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  host_barrier.buffer = output_buffer_.buffer;
  host_barrier.offset = 0;
  host_barrier.size = output_buffer_.size;

  vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &host_barrier, 0, nullptr);

  if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS) {
    vkFreeCommandBuffers(context_.device(), context_.command_pool(), 1, &command_buffer);
    throw std::runtime_error("Failed to end command buffer.");
  }

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffer;

  if (vkQueueSubmit(context_.compute_queue(), 1, &submit_info, VK_NULL_HANDLE) != VK_SUCCESS) {
    vkFreeCommandBuffers(context_.device(), context_.command_pool(), 1, &command_buffer);
    throw std::runtime_error("Failed to submit compute command buffer.");
  }

  if (vkQueueWaitIdle(context_.compute_queue()) != VK_SUCCESS) {
    vkFreeCommandBuffers(context_.device(), context_.command_pool(), 1, &command_buffer);
    throw std::runtime_error("Failed to wait for compute queue idle.");
  }

  vkFreeCommandBuffers(context_.device(), context_.command_pool(), 1, &command_buffer);

  const auto* values = static_cast<const ComplexValue*>(output_buffer_.mapped);
  const size_t value_count = static_cast<size_t>(output_buffer_.size / sizeof(ComplexValue));
  return std::vector<ComplexValue>(values, values + value_count);
}
