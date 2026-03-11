#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>

struct BufferAllocation {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  void* mapped = nullptr;
  VkDeviceSize size = 0;
};

uint32_t FindMemoryType(VkPhysicalDevice physical_device,
                        uint32_t type_filter,
                        VkMemoryPropertyFlags properties);

BufferAllocation CreateBuffer(VkPhysicalDevice physical_device,
                              VkDevice device,
                              VkDeviceSize size,
                              VkBufferUsageFlags usage,
                              VkMemoryPropertyFlags properties);

void DestroyBuffer(VkDevice device, BufferAllocation& allocation);
