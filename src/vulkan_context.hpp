#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

struct DeviceInfo {
  std::string name;
  std::string driver_name;
  std::string driver_info;
  uint32_t vendor_id = 0;
  uint32_t device_id = 0;
  uint32_t api_version = 0;
  uint32_t driver_version = 0;
  bool has_portability_subset = false;
  bool has_scalar_block_layout = false;
};

class VulkanContext {
 public:
  explicit VulkanContext(uint32_t requested_device_index);
  ~VulkanContext();

  VulkanContext(const VulkanContext&) = delete;
  VulkanContext& operator=(const VulkanContext&) = delete;

  static std::vector<DeviceInfo> EnumerateDevices();

  VkInstance instance() const { return instance_; }
  VkPhysicalDevice physical_device() const { return physical_device_; }
  VkDevice device() const { return device_; }
  VkQueue compute_queue() const { return compute_queue_; }
  uint32_t queue_family_index() const { return queue_family_index_; }
  VkCommandPool command_pool() const { return command_pool_; }
  const DeviceInfo& selected_device_info() const { return selected_device_info_; }
  uint32_t loader_api_version() const { return loader_api_version_; }

 private:
  void CreateInstance();
  void PickPhysicalDevice(uint32_t requested_device_index);
  void CreateDevice();
  void CreateCommandPool();

  VkInstance instance_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue compute_queue_ = VK_NULL_HANDLE;
  VkCommandPool command_pool_ = VK_NULL_HANDLE;
  uint32_t queue_family_index_ = 0;
  uint32_t loader_api_version_ = VK_API_VERSION_1_0;
  DeviceInfo selected_device_info_;
  bool enable_portability_enumeration_ = false;
  bool enable_portability_subset_ = false;
};
