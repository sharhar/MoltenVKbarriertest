#include "buffer_utils.hpp"

#include <stdexcept>

uint32_t FindMemoryType(VkPhysicalDevice physical_device,
                        uint32_t type_filter,
                        VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties mem_properties{};
  vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

  for (uint32_t i = 0; i < mem_properties.memoryTypeCount; ++i) {
    const bool type_matches = (type_filter & (1u << i)) != 0;
    const bool property_matches =
        (mem_properties.memoryTypes[i].propertyFlags & properties) == properties;
    if (type_matches && property_matches) {
      return i;
    }
  }

  throw std::runtime_error("Failed to find a suitable Vulkan memory type.");
}

BufferAllocation CreateBuffer(VkPhysicalDevice physical_device,
                              VkDevice device,
                              VkDeviceSize size,
                              VkBufferUsageFlags usage,
                              VkMemoryPropertyFlags properties) {
  BufferAllocation allocation{};
  allocation.size = size;

  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = size;
  buffer_info.usage = usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateBuffer(device, &buffer_info, nullptr, &allocation.buffer) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create Vulkan buffer.");
  }

  VkMemoryRequirements mem_requirements{};
  vkGetBufferMemoryRequirements(device, allocation.buffer, &mem_requirements);

  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.allocationSize = mem_requirements.size;
  alloc_info.memoryTypeIndex =
      FindMemoryType(physical_device, mem_requirements.memoryTypeBits, properties);

  if (vkAllocateMemory(device, &alloc_info, nullptr, &allocation.memory) != VK_SUCCESS) {
    vkDestroyBuffer(device, allocation.buffer, nullptr);
    allocation.buffer = VK_NULL_HANDLE;
    throw std::runtime_error("Failed to allocate Vulkan buffer memory.");
  }

  vkBindBufferMemory(device, allocation.buffer, allocation.memory, 0);

  if ((properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
    if (vkMapMemory(device, allocation.memory, 0, size, 0, &allocation.mapped) != VK_SUCCESS) {
      DestroyBuffer(device, allocation);
      throw std::runtime_error("Failed to map Vulkan buffer memory.");
    }
  }

  return allocation;
}

void DestroyBuffer(VkDevice device, BufferAllocation& allocation) {
  if (allocation.mapped != nullptr) {
    vkUnmapMemory(device, allocation.memory);
    allocation.mapped = nullptr;
  }
  if (allocation.memory != VK_NULL_HANDLE) {
    vkFreeMemory(device, allocation.memory, nullptr);
    allocation.memory = VK_NULL_HANDLE;
  }
  if (allocation.buffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(device, allocation.buffer, nullptr);
    allocation.buffer = VK_NULL_HANDLE;
  }
  allocation.size = 0;
}
