#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#ifndef VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME
#define VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME "VK_KHR_portability_subset"
#endif

constexpr std::size_t kFftCount = 875;
constexpr std::size_t kFftLength = 125;
constexpr std::size_t kThreadsPerWorkgroup = 25;
constexpr int kDefaultIterations = 10;
constexpr float kDefaultAbsTolerance = 5e-3f;
constexpr float kDefaultRelTolerance = 5e-4f;
constexpr std::size_t kMismatchPreviewLimit = 5;

struct Complex32 {
    float real;
    float imag;
};

struct Mismatch {
    std::size_t index;
    Complex32 expected;
    Complex32 actual;
    float absError;
};

struct IterationResult {
    std::size_t mismatchCount;
    std::vector<Mismatch> mismatches;
    float maxAbsError;
};

struct Variant {
    std::string title;
    std::string shaderPath;
};

struct Config {
    std::string inputPath = "../data/fft_875x125_input.bin";
    std::string referencePath = "../data/fft_875x125_reference.bin";
    std::string barrierOnlyShaderPath = "barrier_only.spv";
    std::string memoryBarrierShaderPath = "memory_barrier_then_barrier.spv";
    std::string shaderDumpDir = "shader_dump";
    int iterations = kDefaultIterations;
    float absTolerance = kDefaultAbsTolerance;
    float relTolerance = kDefaultRelTolerance;
};

class ScopedShaderModule {
  public:
    ScopedShaderModule() = default;

    ScopedShaderModule(VkDevice device, VkShaderModule module) : device_(device), module_(module) {}

    ScopedShaderModule(const ScopedShaderModule&) = delete;
    ScopedShaderModule& operator=(const ScopedShaderModule&) = delete;

    ScopedShaderModule(ScopedShaderModule&& other) noexcept : device_(other.device_), module_(other.module_) {
        other.device_ = VK_NULL_HANDLE;
        other.module_ = VK_NULL_HANDLE;
    }

    ScopedShaderModule& operator=(ScopedShaderModule&& other) noexcept {
        if (this != &other) {
            reset();
            device_ = other.device_;
            module_ = other.module_;
            other.device_ = VK_NULL_HANDLE;
            other.module_ = VK_NULL_HANDLE;
        }
        return *this;
    }

    ~ScopedShaderModule() { reset(); }

    VkShaderModule get() const { return module_; }

  private:
    void reset() {
        if (module_ != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device_, module_, nullptr);
        }
        module_ = VK_NULL_HANDLE;
        device_ = VK_NULL_HANDLE;
    }

    VkDevice device_ = VK_NULL_HANDLE;
    VkShaderModule module_ = VK_NULL_HANDLE;
};

class ScopedPipeline {
  public:
    ScopedPipeline() = default;

    ScopedPipeline(VkDevice device, VkPipeline pipeline) : device_(device), pipeline_(pipeline) {}

    ScopedPipeline(const ScopedPipeline&) = delete;
    ScopedPipeline& operator=(const ScopedPipeline&) = delete;

    ScopedPipeline(ScopedPipeline&& other) noexcept : device_(other.device_), pipeline_(other.pipeline_) {
        other.device_ = VK_NULL_HANDLE;
        other.pipeline_ = VK_NULL_HANDLE;
    }

    ScopedPipeline& operator=(ScopedPipeline&& other) noexcept {
        if (this != &other) {
            reset();
            device_ = other.device_;
            pipeline_ = other.pipeline_;
            other.device_ = VK_NULL_HANDLE;
            other.pipeline_ = VK_NULL_HANDLE;
        }
        return *this;
    }

    ~ScopedPipeline() { reset(); }

    VkPipeline get() const { return pipeline_; }

  private:
    void reset() {
        if (pipeline_ != VK_NULL_HANDLE) {
            vkDestroyPipeline(device_, pipeline_, nullptr);
        }
        pipeline_ = VK_NULL_HANDLE;
        device_ = VK_NULL_HANDLE;
    }

    VkDevice device_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

class ScopedCommandBuffer {
  public:
    ScopedCommandBuffer() = default;

    ScopedCommandBuffer(VkDevice device, VkCommandPool pool, VkCommandBuffer buffer)
        : device_(device), pool_(pool), buffer_(buffer) {}

    ScopedCommandBuffer(const ScopedCommandBuffer&) = delete;
    ScopedCommandBuffer& operator=(const ScopedCommandBuffer&) = delete;

    ScopedCommandBuffer(ScopedCommandBuffer&& other) noexcept
        : device_(other.device_), pool_(other.pool_), buffer_(other.buffer_) {
        other.device_ = VK_NULL_HANDLE;
        other.pool_ = VK_NULL_HANDLE;
        other.buffer_ = VK_NULL_HANDLE;
    }

    ScopedCommandBuffer& operator=(ScopedCommandBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            device_ = other.device_;
            pool_ = other.pool_;
            buffer_ = other.buffer_;
            other.device_ = VK_NULL_HANDLE;
            other.pool_ = VK_NULL_HANDLE;
            other.buffer_ = VK_NULL_HANDLE;
        }
        return *this;
    }

    ~ScopedCommandBuffer() { reset(); }

    VkCommandBuffer get() const { return buffer_; }

  private:
    void reset() {
        if (buffer_ != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(device_, pool_, 1, &buffer_);
        }
        buffer_ = VK_NULL_HANDLE;
        pool_ = VK_NULL_HANDLE;
        device_ = VK_NULL_HANDLE;
    }

    VkDevice device_ = VK_NULL_HANDLE;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer buffer_ = VK_NULL_HANDLE;
};

struct VulkanContext {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamilyIndex = 0;
    VkBuffer storageBuffer = VK_NULL_HANDLE;
    VkDeviceMemory storageMemory = VK_NULL_HANDLE;
    void* mappedMemory = nullptr;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool hostCoherent = false;
    VkDeviceSize nonCoherentAtomSize = 1;

    ~VulkanContext() {
        if (device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device);
        }
        if (mappedMemory != nullptr && device != VK_NULL_HANDLE && storageMemory != VK_NULL_HANDLE) {
            vkUnmapMemory(device, storageMemory);
        }
        if (fence != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
            vkDestroyFence(device, fence, nullptr);
        }
        if (commandPool != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, commandPool, nullptr);
        }
        if (descriptorPool != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        }
        if (pipelineLayout != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        }
        if (descriptorSetLayout != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        }
        if (storageBuffer != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, storageBuffer, nullptr);
        }
        if (storageMemory != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
            vkFreeMemory(device, storageMemory, nullptr);
        }
        if (device != VK_NULL_HANDLE) {
            vkDestroyDevice(device, nullptr);
        }
        if (instance != VK_NULL_HANDLE) {
            vkDestroyInstance(instance, nullptr);
        }
    }
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

const char* vkResultName(VkResult result) {
    switch (result) {
    case VK_SUCCESS:
        return "VK_SUCCESS";
    case VK_ERROR_INCOMPATIBLE_DRIVER:
        return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_EXTENSION_NOT_PRESENT:
        return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_INITIALIZATION_FAILED:
        return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_OUT_OF_HOST_MEMORY:
        return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
        return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_DEVICE_LOST:
        return "VK_ERROR_DEVICE_LOST";
    default:
        return "VK_ERROR_UNKNOWN";
    }
}

void checkVk(VkResult result, const std::string& operation) {
    if (result != VK_SUCCESS) {
        fail(operation + " failed with " + std::string(vkResultName(result)) + " (" + std::to_string(result) + ").");
    }
}

std::string usage() {
    return
        "Usage: ./barrier_test.exec [--input PATH] [--reference PATH] [--iterations N] "
        "[--abs-tol VALUE] [--rel-tol VALUE] [--barrier-only-spv PATH] "
        "[--memory-and-barrier-spv PATH] [--shader-dump-dir PATH]\n\n"
        "Defaults:\n"
        "  --input ../data/fft_875x125_input.bin\n"
        "  --reference ../data/fft_875x125_reference.bin\n"
        "  --iterations 10\n"
        "  --abs-tol 0.005\n"
        "  --rel-tol 0.0005\n"
        "  --barrier-only-spv barrier_only.spv\n"
        "  --memory-and-barrier-spv memory_barrier_then_barrier.spv\n"
        "  --shader-dump-dir ./shader_dump\n";
}

Config parseArgs(int argc, char** argv) {
    Config config;
    std::vector<std::string> args(argv + 1, argv + argc);

    for (std::size_t index = 0; index < args.size(); ++index) {
        const std::string& flag = args[index];
        auto nextValue = [&]() -> const std::string& {
            if (index + 1 >= args.size()) {
                fail("Missing value for " + flag + ".\n\n" + usage());
            }
            ++index;
            return args[index];
        };

        if (flag == "--help" || flag == "-h") {
            std::cout << usage();
            std::exit(0);
        }
        if (flag == "--input") {
            config.inputPath = nextValue();
            continue;
        }
        if (flag == "--reference") {
            config.referencePath = nextValue();
            continue;
        }
        if (flag == "--barrier-only-spv") {
            config.barrierOnlyShaderPath = nextValue();
            continue;
        }
        if (flag == "--memory-and-barrier-spv") {
            config.memoryBarrierShaderPath = nextValue();
            continue;
        }
        if (flag == "--shader-dump-dir") {
            config.shaderDumpDir = nextValue();
            continue;
        }
        if (flag == "--iterations") {
            config.iterations = std::stoi(nextValue());
            if (config.iterations <= 0) {
                fail("Invalid value for --iterations.\n\n" + usage());
            }
            continue;
        }
        if (flag == "--abs-tol") {
            config.absTolerance = std::stof(nextValue());
            if (config.absTolerance < 0.0f) {
                fail("Invalid value for --abs-tol.\n\n" + usage());
            }
            continue;
        }
        if (flag == "--rel-tol") {
            config.relTolerance = std::stof(nextValue());
            if (config.relTolerance < 0.0f) {
                fail("Invalid value for --rel-tol.\n\n" + usage());
            }
            continue;
        }
        fail("Unknown argument: " + flag + "\n\n" + usage());
    }

    return config;
}

std::string configureShaderDumpDir(const std::string& configuredPath) {
    namespace fs = std::filesystem;

    const fs::path dumpPath = fs::absolute(fs::path(configuredPath));
    std::error_code errorCode;
    fs::create_directories(dumpPath, errorCode);
    if (errorCode) {
        fail("Failed to create MoltenVK shader dump directory " + dumpPath.string() + ": " + errorCode.message() +
             ".");
    }

    if (setenv("MVK_CONFIG_SHADER_DUMP_DIR", dumpPath.c_str(), 1) != 0) {
        fail("Failed to set MVK_CONFIG_SHADER_DUMP_DIR to " + dumpPath.string() + ".");
    }

    return dumpPath.string();
}

std::size_t expectedByteCount() {
    return kFftCount * kFftLength * sizeof(Complex32);
}

std::vector<std::uint8_t> loadBinaryFile(const std::string& path, const std::string& missingHint) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        fail(missingHint + ": " + path);
    }

    const std::streamsize size = stream.tellg();
    if (size < 0) {
        fail("Failed to determine file size for " + path + ".");
    }
    stream.seekg(0, std::ios::beg);

    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    if (!stream.read(reinterpret_cast<char*>(bytes.data()), size)) {
        fail("Failed to read file " + path + ".");
    }
    return bytes;
}

std::vector<Complex32> decodeComplexBlob(const std::vector<std::uint8_t>& bytes, const std::string& path) {
    const std::size_t expectedBytes = expectedByteCount();
    if (bytes.size() != expectedBytes) {
        fail("Blob size mismatch for " + path + ". Expected " + std::to_string(expectedBytes) +
             " bytes, got " + std::to_string(bytes.size()) + ".");
    }

    std::vector<Complex32> values(bytes.size() / sizeof(Complex32));
    std::memcpy(values.data(), bytes.data(), bytes.size());
    return values;
}

float toleranceLimit(const Complex32& expected, float absTolerance, float relTolerance) {
    const float magnitude = std::hypot(expected.real, expected.imag);
    return std::max(absTolerance, magnitude * relTolerance);
}

IterationResult evaluateResults(const Complex32* actual,
                                const std::vector<Complex32>& reference,
                                float absTolerance,
                                float relTolerance) {
    IterationResult result{};
    result.maxAbsError = 0.0f;
    result.mismatches.reserve(kMismatchPreviewLimit);

    for (std::size_t index = 0; index < reference.size(); ++index) {
        const Complex32& expectedValue = reference[index];
        const Complex32& actualValue = actual[index];
        const float dx = actualValue.real - expectedValue.real;
        const float dy = actualValue.imag - expectedValue.imag;
        const float absError = std::hypot(dx, dy);

        result.maxAbsError = std::max(result.maxAbsError, absError);
        if (absError > toleranceLimit(expectedValue, absTolerance, relTolerance)) {
            ++result.mismatchCount;
            if (result.mismatches.size() < kMismatchPreviewLimit) {
                result.mismatches.push_back(Mismatch{index, expectedValue, actualValue, absError});
            }
        }
    }

    return result;
}

std::uint32_t makeVersionStringComponent(std::uint32_t major,
                                         std::uint32_t minor,
                                         std::uint32_t patch) {
    return VK_MAKE_API_VERSION(0, major, minor, patch);
}

std::string formatApiVersion(std::uint32_t version) {
    return std::to_string(VK_API_VERSION_MAJOR(version)) + "." +
           std::to_string(VK_API_VERSION_MINOR(version)) + "." +
           std::to_string(VK_API_VERSION_PATCH(version));
}

bool hasExtension(const std::vector<VkExtensionProperties>& extensions, const char* name) {
    for (const VkExtensionProperties& extension : extensions) {
        if (std::strcmp(extension.extensionName, name) == 0) {
            return true;
        }
    }
    return false;
}

uint32_t findMemoryTypeIndex(VkPhysicalDevice physicalDevice,
                             uint32_t memoryTypeBits,
                             VkMemoryPropertyFlags required,
                             VkMemoryPropertyFlags preferred,
                             bool* hostCoherent) {
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

    const auto matches = [&](VkMemoryPropertyFlags flags, VkMemoryPropertyFlags wanted) {
        return (flags & wanted) == wanted;
    };

    for (uint32_t index = 0; index < memoryProperties.memoryTypeCount; ++index) {
        if ((memoryTypeBits & (1u << index)) == 0u) {
            continue;
        }

        const VkMemoryPropertyFlags flags = memoryProperties.memoryTypes[index].propertyFlags;
        if (matches(flags, required | preferred)) {
            *hostCoherent = matches(flags, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            return index;
        }
    }

    for (uint32_t index = 0; index < memoryProperties.memoryTypeCount; ++index) {
        if ((memoryTypeBits & (1u << index)) == 0u) {
            continue;
        }

        const VkMemoryPropertyFlags flags = memoryProperties.memoryTypes[index].propertyFlags;
        if (matches(flags, required)) {
            *hostCoherent = matches(flags, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            return index;
        }
    }

    fail("Failed to find a host-visible Vulkan memory type for the storage buffer.");
}

void flushMappedMemoryIfNeeded(const VulkanContext& context) {
    if (context.hostCoherent) {
        return;
    }

    VkMappedMemoryRange range{};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = context.storageMemory;
    range.offset = 0;
    range.size = VK_WHOLE_SIZE;
    checkVk(vkFlushMappedMemoryRanges(context.device, 1, &range), "vkFlushMappedMemoryRanges");
}

void invalidateMappedMemoryIfNeeded(const VulkanContext& context) {
    if (context.hostCoherent) {
        return;
    }

    VkMappedMemoryRange range{};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = context.storageMemory;
    range.offset = 0;
    range.size = VK_WHOLE_SIZE;
    checkVk(vkInvalidateMappedMemoryRanges(context.device, 1, &range), "vkInvalidateMappedMemoryRanges");
}

VkPhysicalDevice pickPhysicalDevice(VkInstance instance, uint32_t* queueFamilyIndex) {
    uint32_t deviceCount = 0;
    checkVk(vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr), "vkEnumeratePhysicalDevices(count)");
    if (deviceCount == 0) {
        fail("No Vulkan physical devices were found. On macOS this usually means MoltenVK is unavailable.");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    checkVk(vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data()),
            "vkEnumeratePhysicalDevices(list)");

    for (VkPhysicalDevice device : devices) {
        uint32_t familyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &familyCount, nullptr);
        std::vector<VkQueueFamilyProperties> families(familyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &familyCount, families.data());

        for (uint32_t familyIndex = 0; familyIndex < familyCount; ++familyIndex) {
            if ((families[familyIndex].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0u) {
                *queueFamilyIndex = familyIndex;
                return device;
            }
        }
    }

    fail("No Vulkan queue family with compute support was found.");
}

VulkanContext createContext(std::size_t bufferByteCount) {
    VulkanContext context;

    uint32_t instanceExtensionCount = 0;
    checkVk(vkEnumerateInstanceExtensionProperties(nullptr, &instanceExtensionCount, nullptr),
            "vkEnumerateInstanceExtensionProperties(count)");
    std::vector<VkExtensionProperties> instanceExtensions(instanceExtensionCount);
    checkVk(vkEnumerateInstanceExtensionProperties(nullptr, &instanceExtensionCount, instanceExtensions.data()),
            "vkEnumerateInstanceExtensionProperties(list)");

    std::vector<const char*> enabledInstanceExtensions;
    VkInstanceCreateFlags instanceFlags = 0;
    if (hasExtension(instanceExtensions, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        enabledInstanceExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        instanceFlags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }

    VkApplicationInfo applicationInfo{};
    applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    applicationInfo.pApplicationName = "MoltenVK FFT Barrier Repro";
    applicationInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    applicationInfo.pEngineName = "None";
    applicationInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    applicationInfo.apiVersion = makeVersionStringComponent(1, 1, 0);

    VkInstanceCreateInfo instanceCreateInfo{};
    instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceCreateInfo.flags = instanceFlags;
    instanceCreateInfo.pApplicationInfo = &applicationInfo;
    instanceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(enabledInstanceExtensions.size());
    instanceCreateInfo.ppEnabledExtensionNames = enabledInstanceExtensions.data();

    checkVk(vkCreateInstance(&instanceCreateInfo, nullptr, &context.instance), "vkCreateInstance");

    context.physicalDevice = pickPhysicalDevice(context.instance, &context.queueFamilyIndex);

    VkPhysicalDeviceProperties deviceProperties{};
    vkGetPhysicalDeviceProperties(context.physicalDevice, &deviceProperties);
    context.nonCoherentAtomSize = deviceProperties.limits.nonCoherentAtomSize;

    uint32_t deviceExtensionCount = 0;
    checkVk(vkEnumerateDeviceExtensionProperties(context.physicalDevice, nullptr, &deviceExtensionCount, nullptr),
            "vkEnumerateDeviceExtensionProperties(count)");
    std::vector<VkExtensionProperties> deviceExtensions(deviceExtensionCount);
    checkVk(vkEnumerateDeviceExtensionProperties(context.physicalDevice, nullptr, &deviceExtensionCount,
                                                 deviceExtensions.data()),
            "vkEnumerateDeviceExtensionProperties(list)");

    std::vector<const char*> enabledDeviceExtensions;
    if (hasExtension(deviceExtensions, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME)) {
        enabledDeviceExtensions.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
    }

    const float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = context.queueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(enabledDeviceExtensions.size());
    deviceCreateInfo.ppEnabledExtensionNames = enabledDeviceExtensions.data();

    checkVk(vkCreateDevice(context.physicalDevice, &deviceCreateInfo, nullptr, &context.device), "vkCreateDevice");
    vkGetDeviceQueue(context.device, context.queueFamilyIndex, 0, &context.queue);

    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = bufferByteCount;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    checkVk(vkCreateBuffer(context.device, &bufferCreateInfo, nullptr, &context.storageBuffer), "vkCreateBuffer");

    VkMemoryRequirements memoryRequirements{};
    vkGetBufferMemoryRequirements(context.device, context.storageBuffer, &memoryRequirements);

    const uint32_t memoryTypeIndex = findMemoryTypeIndex(
        context.physicalDevice,
        memoryRequirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &context.hostCoherent);

    VkMemoryAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.allocationSize = memoryRequirements.size;
    allocateInfo.memoryTypeIndex = memoryTypeIndex;
    checkVk(vkAllocateMemory(context.device, &allocateInfo, nullptr, &context.storageMemory), "vkAllocateMemory");
    checkVk(vkBindBufferMemory(context.device, context.storageBuffer, context.storageMemory, 0), "vkBindBufferMemory");
    checkVk(vkMapMemory(context.device, context.storageMemory, 0, bufferByteCount, 0, &context.mappedMemory),
            "vkMapMemory");

    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutInfo{};
    descriptorSetLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutInfo.bindingCount = 1;
    descriptorSetLayoutInfo.pBindings = &binding;
    checkVk(vkCreateDescriptorSetLayout(context.device, &descriptorSetLayoutInfo, nullptr, &context.descriptorSetLayout),
            "vkCreateDescriptorSetLayout");

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &context.descriptorSetLayout;
    checkVk(vkCreatePipelineLayout(context.device, &pipelineLayoutInfo, nullptr, &context.pipelineLayout),
            "vkCreatePipelineLayout");

    VkDescriptorPoolSize descriptorPoolSize{};
    descriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorPoolSize.descriptorCount = 1;

    VkDescriptorPoolCreateInfo descriptorPoolInfo{};
    descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolInfo.maxSets = 1;
    descriptorPoolInfo.poolSizeCount = 1;
    descriptorPoolInfo.pPoolSizes = &descriptorPoolSize;
    checkVk(vkCreateDescriptorPool(context.device, &descriptorPoolInfo, nullptr, &context.descriptorPool),
            "vkCreateDescriptorPool");

    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo{};
    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.descriptorPool = context.descriptorPool;
    descriptorSetAllocateInfo.descriptorSetCount = 1;
    descriptorSetAllocateInfo.pSetLayouts = &context.descriptorSetLayout;
    checkVk(vkAllocateDescriptorSets(context.device, &descriptorSetAllocateInfo, &context.descriptorSet),
            "vkAllocateDescriptorSets");

    VkDescriptorBufferInfo descriptorBufferInfo{};
    descriptorBufferInfo.buffer = context.storageBuffer;
    descriptorBufferInfo.offset = 0;
    descriptorBufferInfo.range = bufferByteCount;

    VkWriteDescriptorSet descriptorWrite{};
    descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrite.dstSet = context.descriptorSet;
    descriptorWrite.dstBinding = 0;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrite.pBufferInfo = &descriptorBufferInfo;
    vkUpdateDescriptorSets(context.device, 1, &descriptorWrite, 0, nullptr);

    VkCommandPoolCreateInfo commandPoolInfo{};
    commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolInfo.queueFamilyIndex = context.queueFamilyIndex;
    checkVk(vkCreateCommandPool(context.device, &commandPoolInfo, nullptr, &context.commandPool),
            "vkCreateCommandPool");

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    checkVk(vkCreateFence(context.device, &fenceInfo, nullptr, &context.fence), "vkCreateFence");

    return context;
}

std::vector<std::uint32_t> loadSpirvWords(const std::string& path) {
    std::vector<std::uint8_t> bytes =
        loadBinaryFile(path, "Failed to load shader binary. Build the project first with ./build.sh");
    if ((bytes.size() % sizeof(std::uint32_t)) != 0u) {
        fail("SPIR-V binary size is not a multiple of 4 bytes: " + path);
    }

    std::vector<std::uint32_t> words(bytes.size() / sizeof(std::uint32_t));
    std::memcpy(words.data(), bytes.data(), bytes.size());
    return words;
}

ScopedShaderModule createShaderModule(VkDevice device, const std::vector<std::uint32_t>& words) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = words.size() * sizeof(std::uint32_t);
    createInfo.pCode = words.data();

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    checkVk(vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule), "vkCreateShaderModule");
    return ScopedShaderModule(device, shaderModule);
}

ScopedPipeline createPipeline(VkDevice device,
                              VkPipelineLayout pipelineLayout,
                              const ScopedShaderModule& shaderModule) {
    VkPipelineShaderStageCreateInfo shaderStage{};
    shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStage.module = shaderModule.get();
    shaderStage.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = shaderStage;
    pipelineInfo.layout = pipelineLayout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    checkVk(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline),
            "vkCreateComputePipelines");
    return ScopedPipeline(device, pipeline);
}

ScopedCommandBuffer recordCommandBuffer(const VulkanContext& context, VkPipeline pipeline) {
    VkCommandBufferAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocateInfo.commandPool = context.commandPool;
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    checkVk(vkAllocateCommandBuffers(context.device, &allocateInfo, &commandBuffer), "vkAllocateCommandBuffers");

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    checkVk(vkBeginCommandBuffer(commandBuffer, &beginInfo), "vkBeginCommandBuffer");

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(commandBuffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            context.pipelineLayout,
                            0,
                            1,
                            &context.descriptorSet,
                            0,
                            nullptr);
    vkCmdDispatch(commandBuffer, static_cast<uint32_t>(kFftCount), 1, 1);

    VkBufferMemoryBarrier hostReadBarrier{};
    hostReadBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    hostReadBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    hostReadBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    hostReadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    hostReadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    hostReadBarrier.buffer = context.storageBuffer;
    hostReadBarrier.offset = 0;
    hostReadBarrier.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(commandBuffer,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT,
                         0,
                         0,
                         nullptr,
                         1,
                         &hostReadBarrier,
                         0,
                         nullptr);

    checkVk(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");
    return ScopedCommandBuffer(context.device, context.commandPool, commandBuffer);
}

void printIterationResult(const IterationResult& result, int iteration, std::size_t totalCount) {
    const char* status = result.mismatchCount == 0 ? "PASS" : "FAIL";
    std::cout << "  Iteration " << iteration << ": " << result.mismatchCount << " mismatches out of " << totalCount
              << " (" << status << "), max abs error " << result.maxAbsError << "\n";

    for (const Mismatch& mismatch : result.mismatches) {
        const std::size_t fftIndex = mismatch.index / kFftLength;
        const std::size_t binIndex = mismatch.index % kFftLength;
        std::cout << "    fft " << fftIndex << ", bin " << binIndex << ", index " << mismatch.index << ": expected ("
                  << mismatch.expected.real << ", " << mismatch.expected.imag << "), got (" << mismatch.actual.real
                  << ", " << mismatch.actual.imag << "), abs error " << mismatch.absError << "\n";
    }
}

void printSummary(const std::vector<IterationResult>& results) {
    const auto failingIterations = std::count_if(results.begin(), results.end(), [](const IterationResult& result) {
        return result.mismatchCount > 0;
    });

    if (failingIterations == 0) {
        std::cout << "  Result: PASS (0/" << results.size() << " iterations had mismatches)\n\n";
    } else {
        std::cout << "  Result: FAIL (mismatches in " << failingIterations << "/" << results.size()
                  << " iterations)\n\n";
    }
}

void printConclusion(const std::vector<IterationResult>& barrierOnlyResults,
                     const std::vector<IterationResult>& memoryBarrierResults) {
    const auto barrierOnlyFailures =
        std::count_if(barrierOnlyResults.begin(), barrierOnlyResults.end(), [](const IterationResult& result) {
            return result.mismatchCount > 0;
        });
    const auto memoryBarrierFailures = std::count_if(memoryBarrierResults.begin(),
                                                     memoryBarrierResults.end(),
                                                     [](const IterationResult& result) {
                                                         return result.mismatchCount > 0;
                                                     });

    std::cout << "=== CONCLUSION ===\n";
    if (barrierOnlyFailures > 0 && memoryBarrierFailures == 0) {
        std::cout << "The MoltenVK FFT workload reproduces a synchronization failure with barrier() alone.\n";
        std::cout << "Adding memoryBarrier() before barrier() is sufficient to match the NumPy reference on this "
                     "machine.\n";
        return;
    }

    if (barrierOnlyFailures == 0 && memoryBarrierFailures == 0) {
        std::cout << "This machine did not reproduce the Vulkan barrier bug with the FFT workload.\n";
        std::cout << "That is still useful data and may indicate hardware-, driver-, or OS-specific behavior.\n";
        return;
    }

    std::cout << "Observed failure counts:\n";
    std::cout << "  barrier() only: " << barrierOnlyFailures << "\n";
    std::cout << "  memoryBarrier(); barrier(): " << memoryBarrierFailures << "\n";
}

std::vector<IterationResult> runVariant(const VulkanContext& context,
                                        const Variant& variant,
                                        const std::vector<std::uint8_t>& inputBytes,
                                        const std::vector<Complex32>& reference,
                                        const Config& config) {
    const std::vector<std::uint32_t> spirvWords = loadSpirvWords(variant.shaderPath);
    const ScopedShaderModule shaderModule = createShaderModule(context.device, spirvWords);
    const ScopedPipeline pipeline = createPipeline(context.device, context.pipelineLayout, shaderModule);
    const ScopedCommandBuffer commandBuffer = recordCommandBuffer(context, pipeline.get());

    std::vector<IterationResult> results;
    results.reserve(static_cast<std::size_t>(config.iterations));

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    VkCommandBuffer rawCommandBuffer = commandBuffer.get();
    submitInfo.pCommandBuffers = &rawCommandBuffer;

    for (int iteration = 0; iteration < config.iterations; ++iteration) {
        std::memcpy(context.mappedMemory, inputBytes.data(), inputBytes.size());
        flushMappedMemoryIfNeeded(context);

        checkVk(vkResetFences(context.device, 1, &context.fence), "vkResetFences");
        checkVk(vkQueueSubmit(context.queue, 1, &submitInfo, context.fence), "vkQueueSubmit");
        checkVk(vkWaitForFences(context.device, 1, &context.fence, VK_TRUE, std::numeric_limits<std::uint64_t>::max()),
                "vkWaitForFences");

        invalidateMappedMemoryIfNeeded(context);
        const Complex32* actual = static_cast<const Complex32*>(context.mappedMemory);
        results.push_back(evaluateResults(actual, reference, config.absTolerance, config.relTolerance));
    }

    return results;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Config config = parseArgs(argc, argv);
        const std::string shaderDumpDir = configureShaderDumpDir(config.shaderDumpDir);
        std::cout << "=== Vulkan/MoltenVK FFT Barrier Bug Reproduction ===\n";
        std::cout << "MoltenVK shader dump dir: " << shaderDumpDir << "\n" << std::flush;
        const std::vector<std::uint8_t> inputBytes = loadBinaryFile(
            config.inputPath, "Required input blob not found. Run: python3 generate_blobs.py --output-dir data");
        const std::vector<std::uint8_t> referenceBytes = loadBinaryFile(
            config.referencePath,
            "Required reference blob not found. Run: python3 generate_blobs.py --output-dir data");
        const std::vector<Complex32> input = decodeComplexBlob(inputBytes, config.inputPath);
        const std::vector<Complex32> reference = decodeComplexBlob(referenceBytes, config.referencePath);
        (void)input;

        VulkanContext context = createContext(inputBytes.size());

        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(context.physicalDevice, &properties);

        std::cout << "Device: " << properties.deviceName << "\n";
        std::cout << "Vulkan API: " << formatApiVersion(properties.apiVersion) << "\n";
        std::cout << "Driver version: " << properties.driverVersion << "\n";
        std::cout << "FFT layout: " << kFftCount << " x " << kFftLength << "\n";
        std::cout << "Threads per workgroup: " << kThreadsPerWorkgroup << "\n";
        std::cout << "Iterations per variant: " << config.iterations << "\n";
        std::cout << "Tolerance: abs <= " << config.absTolerance << ", rel <= " << config.relTolerance << "\n";
        std::cout << "Input blob: " << config.inputPath << "\n";
        std::cout << "Reference blob: " << config.referencePath << "\n";
        std::cout << "barrier() shader: " << config.barrierOnlyShaderPath << "\n";
        std::cout << "memoryBarrier(); barrier() shader: " << config.memoryBarrierShaderPath << "\n";
        std::cout << "Expected bytes per blob: " << expectedByteCount() << "\n\n";

        const std::vector<Variant> variants = {
            Variant{"barrier()", config.barrierOnlyShaderPath},
            Variant{"memoryBarrier(); barrier()", config.memoryBarrierShaderPath},
        };

        std::vector<IterationResult> barrierOnlyResults;
        std::vector<IterationResult> memoryBarrierResults;

        for (std::size_t variantIndex = 0; variantIndex < variants.size(); ++variantIndex) {
            const Variant& variant = variants[variantIndex];
            std::cout << "--- Test " << (variantIndex + 1) << ": " << variant.title << " ---\n";
            const std::vector<IterationResult> results =
                runVariant(context, variant, inputBytes, reference, config);

            for (std::size_t iterationIndex = 0; iterationIndex < results.size(); ++iterationIndex) {
                printIterationResult(results[iterationIndex], static_cast<int>(iterationIndex + 1), reference.size());
            }

            printSummary(results);
            if (variantIndex == 0) {
                barrierOnlyResults = results;
            } else {
                memoryBarrierResults = results;
            }
        }

        printConclusion(barrierOnlyResults, memoryBarrierResults);
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "ERROR: " << exception.what() << "\n";
        return 1;
    }
}
