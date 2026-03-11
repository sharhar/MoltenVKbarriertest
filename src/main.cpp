#include "compute_pipeline.hpp"
#include "shader_utils.hpp"
#include "vulkan_context.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
constexpr uint32_t kLocalSizeX = 64;
constexpr uint32_t kValuesPerThread = 8;
constexpr uint32_t kRawSlots = kLocalSizeX * kValuesPerThread;
constexpr uint32_t kPaddedSlots = kRawSlots + (kRawSlots >> 4);

enum class ShaderVariant {
  kBarrierOnly,
  kMemoryBarrierPlusBarrier,
  kGroupMemoryBarrierPlusBarrier,
};

struct Options {
  bool list_devices = false;
  uint32_t device_index = 0;
  ShaderVariant shader_variant = ShaderVariant::kBarrierOnly;
  uint32_t workgroups = 8192;
  uint32_t rounds = 256;
  uint32_t repetitions = 50;
  bool verbose = false;
  bool dump_spirv_info = false;
  uint32_t first_mismatch_limit = 8;
  std::filesystem::path shader_dir = REPRO_DEFAULT_SHADER_DIR;
};

uint32_t MixBits(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

uint32_t PaddedIndex(uint32_t raw_index) {
  return raw_index + (raw_index >> 4);
}

std::string VariantName(ShaderVariant variant) {
  switch (variant) {
    case ShaderVariant::kBarrierOnly:
      return "barrier_only";
    case ShaderVariant::kMemoryBarrierPlusBarrier:
      return "memorybarrier_plus_barrier";
    case ShaderVariant::kGroupMemoryBarrierPlusBarrier:
      return "groupmemorybarrier_plus_barrier";
  }
  return "unknown";
}

std::filesystem::path VariantShaderPath(const std::filesystem::path& shader_dir,
                                        ShaderVariant variant) {
  return shader_dir / (VariantName(variant) + ".comp.spv");
}

std::string Hex32(uint32_t value) {
  std::ostringstream stream;
  stream << "0x" << std::hex << std::setw(8) << std::setfill('0') << value;
  return stream.str();
}

void PrintUsage() {
  std::cout
      << "Usage: moltenvk_barrier_repro [options]\n"
      << "  --list-devices\n"
      << "  --device <index>\n"
      << "  --shader barrier_only|memorybarrier_plus_barrier|groupmemorybarrier_plus_barrier\n"
      << "  --workgroups <N>\n"
      << "  --rounds <N>\n"
      << "  --repetitions <N>\n"
      << "  --shader-dir <path>\n"
      << "  --first-mismatch-limit <N>\n"
      << "  --dump-spirv-info\n"
      << "  --verbose\n";
}

uint32_t ParseUint(std::string_view text, const char* flag_name) {
  size_t consumed = 0;
  const unsigned long value = std::stoul(std::string(text), &consumed, 10);
  if (consumed != text.size() || value > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error(std::string("Invalid value for ") + flag_name + ": " +
                             std::string(text));
  }
  return static_cast<uint32_t>(value);
}

ShaderVariant ParseShaderVariant(std::string_view text) {
  if (text == "barrier_only") {
    return ShaderVariant::kBarrierOnly;
  }
  if (text == "memorybarrier_plus_barrier") {
    return ShaderVariant::kMemoryBarrierPlusBarrier;
  }
  if (text == "groupmemorybarrier_plus_barrier") {
    return ShaderVariant::kGroupMemoryBarrierPlusBarrier;
  }
  throw std::runtime_error("Unknown shader variant: " + std::string(text));
}

Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    auto require_value = [&](const char* flag_name) -> std::string_view {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("Missing value for ") + flag_name);
      }
      ++i;
      return argv[i];
    };

    if (arg == "--help" || arg == "-h") {
      PrintUsage();
      std::exit(0);
    } else if (arg == "--list-devices") {
      options.list_devices = true;
    } else if (arg == "--device") {
      options.device_index = ParseUint(require_value("--device"), "--device");
    } else if (arg == "--shader") {
      options.shader_variant = ParseShaderVariant(require_value("--shader"));
    } else if (arg == "--workgroups") {
      options.workgroups = ParseUint(require_value("--workgroups"), "--workgroups");
    } else if (arg == "--rounds") {
      options.rounds = ParseUint(require_value("--rounds"), "--rounds");
    } else if (arg == "--repetitions") {
      options.repetitions = ParseUint(require_value("--repetitions"), "--repetitions");
    } else if (arg == "--shader-dir") {
      options.shader_dir = require_value("--shader-dir");
    } else if (arg == "--verbose") {
      options.verbose = true;
    } else if (arg == "--dump-spirv-info") {
      options.dump_spirv_info = true;
    } else if (arg == "--first-mismatch-limit") {
      options.first_mismatch_limit =
          ParseUint(require_value("--first-mismatch-limit"), "--first-mismatch-limit");
    } else {
      throw std::runtime_error("Unknown argument: " + std::string(arg));
    }
  }

  if (options.workgroups == 0 || options.rounds == 0 || options.repetitions == 0) {
    throw std::runtime_error("workgroups, rounds, and repetitions must all be greater than zero.");
  }
  return options;
}

std::vector<uint32_t> ComputeReference(const Options& options) {
  const uint64_t total_invocations =
      static_cast<uint64_t>(options.workgroups) * static_cast<uint64_t>(kLocalSizeX);
  std::vector<uint32_t> reference(total_invocations, 0);
  std::array<uint32_t, kPaddedSlots> shared_memory{};

  for (uint32_t group = 0; group < options.workgroups; ++group) {
    std::array<uint32_t, kLocalSizeX> state{};
    for (uint32_t lane = 0; lane < kLocalSizeX; ++lane) {
      state[lane] = MixBits((group + 1u) * 0x9e3779b9U ^ (lane + 1u) * 0x85ebca6bU);
    }

    for (uint32_t round = 0; round < options.rounds; ++round) {
      shared_memory.fill(0);

      for (uint32_t lane = 0; lane < kLocalSizeX; ++lane) {
        for (uint32_t slot = 0; slot < kValuesPerThread; ++slot) {
          const uint32_t write_slot = (slot * 5u + round + (lane >> 4u)) & 7u;
          const uint32_t raw = lane * kValuesPerThread + write_slot;
          const uint32_t value = MixBits(state[lane] ^ (round * 0x27d4eb2dU) ^
                                         (slot * 0x165667b1U) ^ (write_slot * 0x9e3779b9U));
          shared_memory[PaddedIndex(raw)] = value;
        }
      }

      std::array<uint32_t, kLocalSizeX> next_state{};
      for (uint32_t lane = 0; lane < kLocalSizeX; ++lane) {
        uint32_t round_accum = state[lane] ^ (round * 0x94d049bbU);
        for (uint32_t slot = 0; slot < kValuesPerThread; ++slot) {
          const uint32_t src_lane = (lane * 17u + slot * 7u + round * 3u + 13u) & 63u;
          const uint32_t src_slot = (slot * 3u + lane + round) & 7u;
          const uint32_t observed =
              shared_memory[PaddedIndex(src_lane * kValuesPerThread + src_slot)];
          round_accum = MixBits(round_accum ^ observed ^ ((src_lane + 1u) * 0x85ebca6bU) ^
                                ((src_slot + 1u) * 0xc2b2ae35U));
        }
        next_state[lane] = MixBits(round_accum ^ 0x27d4eb2dU ^ lane);
      }
      state = next_state;
    }

    for (uint32_t lane = 0; lane < kLocalSizeX; ++lane) {
      reference[group * kLocalSizeX + lane] = state[lane];
    }
  }

  return reference;
}

void PrintDeviceList(const std::vector<DeviceInfo>& devices) {
  if (devices.empty()) {
    std::cout << "No Vulkan devices found.\n";
    return;
  }

  for (size_t i = 0; i < devices.size(); ++i) {
    const DeviceInfo& device = devices[i];
    std::cout << "[" << i << "] " << device.name << "\n";
    std::cout << "    API version: " << VK_API_VERSION_MAJOR(device.api_version) << "."
              << VK_API_VERSION_MINOR(device.api_version) << "."
              << VK_API_VERSION_PATCH(device.api_version) << "\n";
    std::cout << "    Vendor ID: 0x" << std::hex << device.vendor_id << std::dec
              << " Device ID: 0x" << std::hex << device.device_id << std::dec << "\n";
    std::cout << "    Driver name: "
              << (device.driver_name.empty() ? std::string("<unavailable>") : device.driver_name)
              << "\n";
    std::cout << "    Driver info: "
              << (device.driver_info.empty() ? std::string("<unavailable>") : device.driver_info)
              << "\n";
    std::cout << "    Portability subset: "
              << (device.has_portability_subset ? "present" : "absent") << "\n";
  }
}

void PrintRunHeader(const VulkanContext& context, const Options& options) {
  const DeviceInfo& device = context.selected_device_info();
  std::cout << "Vulkan API version: " << VK_API_VERSION_MAJOR(context.loader_api_version()) << "."
            << VK_API_VERSION_MINOR(context.loader_api_version()) << "."
            << VK_API_VERSION_PATCH(context.loader_api_version()) << "\n";
  std::cout << "Device: " << device.name;
  if (!device.driver_name.empty()) {
    std::cout << " via " << device.driver_name;
  }
  std::cout << "\n";
  std::cout << "Vendor ID: " << Hex32(device.vendor_id) << "\n";
  std::cout << "Device ID: " << Hex32(device.device_id) << "\n";
  std::cout << "Driver version: " << device.driver_version << "\n";
  if (!device.driver_info.empty()) {
    std::cout << "Driver info: " << device.driver_info << "\n";
  }
  std::cout << "Portability subset extension: "
            << (device.has_portability_subset ? "present" : "absent") << "\n";
  std::cout << "Shader variant: " << VariantName(options.shader_variant) << "\n";
  std::cout << "Workgroups: " << options.workgroups << "\n";
  std::cout << "Shader rounds: " << options.rounds << "\n";
  std::cout << "Host repetitions: " << options.repetitions << "\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseOptions(argc, argv);

    if (options.list_devices) {
      PrintDeviceList(VulkanContext::EnumerateDevices());
      return 0;
    }

    VulkanContext context(options.device_index);
    PrintRunHeader(context, options);

    const std::filesystem::path shader_path =
        VariantShaderPath(options.shader_dir, options.shader_variant);
    if (!std::filesystem::exists(shader_path)) {
      throw std::runtime_error("Shader file not found: " + shader_path.string());
    }

    const uint64_t total_output_words =
        static_cast<uint64_t>(options.workgroups) * static_cast<uint64_t>(kLocalSizeX);
    if (total_output_words > std::numeric_limits<uint32_t>::max()) {
      throw std::runtime_error("Requested output size is too large.");
    }

    ComputePipeline pipeline(context, shader_path, static_cast<uint32_t>(total_output_words));

    if (options.dump_spirv_info) {
      std::cout << "Shader path: " << shader_path << "\n";
      std::cout << FormatSpirvSummary(pipeline.spirv_summary()) << "\n";
    }

    const std::vector<uint32_t> reference = ComputeReference(options);

    uint64_t total_mismatches = 0;
    std::optional<uint32_t> first_failing_repetition;
    std::vector<std::string> mismatch_lines;

    for (uint32_t repetition = 0; repetition < options.repetitions; ++repetition) {
      if (options.verbose) {
        std::cout << "Running repetition " << repetition + 1 << "/" << options.repetitions
                  << "\n";
      }

      const std::vector<uint32_t> gpu_output =
          pipeline.Run({options.workgroups, options.rounds}, options.verbose);

      for (size_t index = 0; index < gpu_output.size(); ++index) {
        if (gpu_output[index] != reference[index]) {
          ++total_mismatches;
          if (!first_failing_repetition.has_value()) {
            first_failing_repetition = repetition;
          }
          if (mismatch_lines.size() < options.first_mismatch_limit) {
            std::ostringstream line;
            line << "Mismatch[" << index << "]: expected=" << Hex32(reference[index])
                 << " actual=" << Hex32(gpu_output[index])
                 << " repetition=" << repetition;
            mismatch_lines.push_back(line.str());
          }
        }
      }
    }

    if (first_failing_repetition.has_value()) {
      std::cout << "First failing repetition: " << *first_failing_repetition << "\n";
      for (const std::string& line : mismatch_lines) {
        std::cout << line << "\n";
      }
    }

    std::cout << "Mismatches: " << total_mismatches << "\n";
    std::cout << "Status: " << (total_mismatches == 0 ? "PASS" : "FAIL") << "\n";
    return total_mismatches == 0 ? 0 : 1;
  } catch (const std::exception& exception) {
    std::cerr << "Runtime error: " << exception.what() << "\n";
    return 2;
  }
}
