#include "vulkan_context.hpp"

#include <array>
#include <cstring>
#include <stdexcept>

namespace {
constexpr const char* kPortabilityEnumerationExtension = "VK_KHR_portability_enumeration";
constexpr const char* kGetPhysicalDeviceProperties2Extension =
    "VK_KHR_get_physical_device_properties2";
constexpr const char* kPortabilitySubsetExtension = "VK_KHR_portability_subset";

bool HasExtension(const std::vector<VkExtensionProperties>& properties, const char* name) {
  for (const auto& property : properties) {
    if (std::strcmp(property.extensionName, name) == 0) {
      return true;
    }
  }
  return false;
}

std::vector<VkExtensionProperties> EnumerateInstanceExtensions() {
  uint32_t count = 0;
  vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
  std::vector<VkExtensionProperties> properties(count);
  if (count > 0) {
    vkEnumerateInstanceExtensionProperties(nullptr, &count, properties.data());
  }
  return properties;
}

std::vector<VkPhysicalDevice> EnumeratePhysicalDevices(VkInstance instance) {
  uint32_t count = 0;
  vkEnumeratePhysicalDevices(instance, &count, nullptr);
  std::vector<VkPhysicalDevice> devices(count);
  if (count > 0) {
    vkEnumeratePhysicalDevices(instance, &count, devices.data());
  }
  return devices;
}

DeviceInfo QueryDeviceInfo(VkPhysicalDevice physical_device) {
  VkPhysicalDeviceProperties2 properties2{};
  properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  VkPhysicalDeviceDriverProperties driver_properties{};
  driver_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
  properties2.pNext = &driver_properties;
  vkGetPhysicalDeviceProperties2(physical_device, &properties2);

  uint32_t extension_count = 0;
  vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, nullptr);
  std::vector<VkExtensionProperties> extensions(extension_count);
  if (extension_count > 0) {
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count,
                                         extensions.data());
  }

  DeviceInfo info{};
  info.name = properties2.properties.deviceName;
  info.driver_name = driver_properties.driverName;
  info.driver_info = driver_properties.driverInfo;
  info.vendor_id = properties2.properties.vendorID;
  info.device_id = properties2.properties.deviceID;
  info.api_version = properties2.properties.apiVersion;
  info.driver_version = properties2.properties.driverVersion;
  info.has_portability_subset = HasExtension(extensions, kPortabilitySubsetExtension);
  return info;
}
}  // namespace

VulkanContext::VulkanContext(uint32_t requested_device_index) {
  CreateInstance();
  PickPhysicalDevice(requested_device_index);
  CreateDevice();
  CreateCommandPool();
}

VulkanContext::~VulkanContext() {
  if (command_pool_ != VK_NULL_HANDLE) {
    vkDestroyCommandPool(device_, command_pool_, nullptr);
  }
  if (device_ != VK_NULL_HANDLE) {
    vkDestroyDevice(device_, nullptr);
  }
  if (instance_ != VK_NULL_HANDLE) {
    vkDestroyInstance(instance_, nullptr);
  }
}

std::vector<DeviceInfo> VulkanContext::EnumerateDevices() {
  uint32_t api_version = VK_API_VERSION_1_0;
  if (&vkEnumerateInstanceVersion != nullptr) {
    vkEnumerateInstanceVersion(&api_version);
  }

  const std::vector<VkExtensionProperties> instance_extensions = EnumerateInstanceExtensions();

  std::vector<const char*> enabled_extensions;
  VkInstanceCreateFlags create_flags = 0;
  if (HasExtension(instance_extensions, kPortabilityEnumerationExtension)) {
    enabled_extensions.push_back(kPortabilityEnumerationExtension);
    create_flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
  }
  if (!HasExtension(instance_extensions, kGetPhysicalDeviceProperties2Extension) &&
      VK_API_VERSION_MAJOR(api_version) == 1 && VK_API_VERSION_MINOR(api_version) == 0) {
    enabled_extensions.push_back(kGetPhysicalDeviceProperties2Extension);
  }

  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "moltenvk_barrier_repro";
  app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.pEngineName = "none";
  app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.apiVersion = VK_API_VERSION_1_1;

  VkInstanceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &app_info;
  create_info.enabledExtensionCount = static_cast<uint32_t>(enabled_extensions.size());
  create_info.ppEnabledExtensionNames = enabled_extensions.data();
  create_info.flags = create_flags;

  VkInstance instance = VK_NULL_HANDLE;
  if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create Vulkan instance for device enumeration.");
  }

  std::vector<DeviceInfo> devices;
  for (VkPhysicalDevice physical_device : EnumeratePhysicalDevices(instance)) {
    devices.push_back(QueryDeviceInfo(physical_device));
  }

  vkDestroyInstance(instance, nullptr);
  return devices;
}

void VulkanContext::CreateInstance() {
  if (&vkEnumerateInstanceVersion != nullptr) {
    vkEnumerateInstanceVersion(&loader_api_version_);
  }

  const std::vector<VkExtensionProperties> instance_extensions = EnumerateInstanceExtensions();

  std::vector<const char*> enabled_extensions;
  VkInstanceCreateFlags create_flags = 0;
  if (HasExtension(instance_extensions, kPortabilityEnumerationExtension)) {
    enabled_extensions.push_back(kPortabilityEnumerationExtension);
    create_flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    enable_portability_enumeration_ = true;
  }
  if (!HasExtension(instance_extensions, kGetPhysicalDeviceProperties2Extension) &&
      VK_API_VERSION_MAJOR(loader_api_version_) == 1 &&
      VK_API_VERSION_MINOR(loader_api_version_) == 0) {
    enabled_extensions.push_back(kGetPhysicalDeviceProperties2Extension);
  }

  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "moltenvk_barrier_repro";
  app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.pEngineName = "none";
  app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.apiVersion = VK_API_VERSION_1_1;

  VkInstanceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &app_info;
  create_info.enabledExtensionCount = static_cast<uint32_t>(enabled_extensions.size());
  create_info.ppEnabledExtensionNames = enabled_extensions.data();
  create_info.flags = create_flags;

  if (vkCreateInstance(&create_info, nullptr, &instance_) != VK_SUCCESS) {
    throw std::runtime_error(
        "Failed to create Vulkan instance. Check that the LunarG Vulkan SDK is installed.");
  }
}

void VulkanContext::PickPhysicalDevice(uint32_t requested_device_index) {
  const std::vector<VkPhysicalDevice> physical_devices = EnumeratePhysicalDevices(instance_);
  if (physical_devices.empty()) {
    throw std::runtime_error("No Vulkan physical devices were found.");
  }
  if (requested_device_index >= physical_devices.size()) {
    throw std::runtime_error("Requested device index is out of range.");
  }

  physical_device_ = physical_devices[requested_device_index];
  selected_device_info_ = QueryDeviceInfo(physical_device_);

  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, nullptr);
  std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count,
                                           queue_families.data());

  bool found_compute_queue = false;
  for (uint32_t i = 0; i < queue_family_count; ++i) {
    if ((queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
      queue_family_index_ = i;
      found_compute_queue = true;
      break;
    }
  }

  if (!found_compute_queue) {
    throw std::runtime_error("Selected device does not expose a compute queue family.");
  }

  enable_portability_subset_ = selected_device_info_.has_portability_subset;
}

void VulkanContext::CreateDevice() {
  constexpr float kQueuePriority = 1.0f;

  VkDeviceQueueCreateInfo queue_create_info{};
  queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_create_info.queueFamilyIndex = queue_family_index_;
  queue_create_info.queueCount = 1;
  queue_create_info.pQueuePriorities = &kQueuePriority;

  std::vector<const char*> enabled_extensions;
  if (enable_portability_subset_) {
    enabled_extensions.push_back(kPortabilitySubsetExtension);
  }

  VkDeviceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  create_info.queueCreateInfoCount = 1;
  create_info.pQueueCreateInfos = &queue_create_info;
  create_info.enabledExtensionCount = static_cast<uint32_t>(enabled_extensions.size());
  create_info.ppEnabledExtensionNames = enabled_extensions.data();

  if (vkCreateDevice(physical_device_, &create_info, nullptr, &device_) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create Vulkan logical device.");
  }

  vkGetDeviceQueue(device_, queue_family_index_, 0, &compute_queue_);
}

void VulkanContext::CreateCommandPool() {
  VkCommandPoolCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  create_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  create_info.queueFamilyIndex = queue_family_index_;

  if (vkCreateCommandPool(device_, &create_info, nullptr, &command_pool_) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create command pool.");
  }
}
