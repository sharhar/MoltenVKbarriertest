#include "compute_pipeline.hpp"
#include "shader_utils.hpp"
#include "vulkan_context.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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
constexpr uint32_t kFftBatchCount = 875;
constexpr uint32_t kFftLength = 125;
constexpr uint32_t kLocalSizeX = 25;
constexpr uint32_t kComplexValueCount = kFftBatchCount * kFftLength;

enum class ShaderVariant {
  kBarrierOnly,
  kMemoryBarrierPlusBarrier,
  kGroupMemoryBarrierPlusBarrier,
};

struct Options {
  bool list_devices = false;
  uint32_t device_index = 0;
  ShaderVariant shader_variant = ShaderVariant::kBarrierOnly;
  uint32_t repetitions = 50;
  bool verbose = false;
  bool dump_spirv_info = false;
  uint32_t first_mismatch_limit = 8;
  float abs_tolerance = 1.0e-5f;
  float rel_tolerance = 1.0e-5f;
  std::filesystem::path shader_dir = REPRO_DEFAULT_SHADER_DIR;
  std::filesystem::path input_path = "data/fft_875x125_input.bin";
  std::filesystem::path reference_path = "data/fft_875x125_reference.bin";
};

struct ComparisonStats {
  uint64_t mismatches = 0;
  float max_abs_diff = 0.0f;
  float max_rel_diff = 0.0f;
  size_t max_abs_index = 0;
  size_t max_rel_index = 0;
};

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

std::string FormatComplex(const ComplexValue& value) {
  std::ostringstream stream;
  stream << std::scientific << std::setprecision(8) << "(" << value.real << ", " << value.imag
         << ")";
  return stream.str();
}

void PrintUsage() {
  std::cout
      << "Usage: moltenvk_barrier_repro [options]\n"
      << "  --list-devices\n"
      << "  --device <index>\n"
      << "  --shader barrier_only|memorybarrier_plus_barrier|groupmemorybarrier_plus_barrier\n"
      << "  --input <path>\n"
      << "  --reference <path>\n"
      << "  --repetitions <N>\n"
      << "  --abs-tolerance <float>\n"
      << "  --rel-tolerance <float>\n"
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

float ParseFloat(std::string_view text, const char* flag_name) {
  size_t consumed = 0;
  const float value = std::stof(std::string(text), &consumed);
  if (consumed != text.size()) {
    throw std::runtime_error(std::string("Invalid value for ") + flag_name + ": " +
                             std::string(text));
  }
  return value;
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
    } else if (arg == "--input") {
      options.input_path = require_value("--input");
    } else if (arg == "--reference") {
      options.reference_path = require_value("--reference");
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
    } else if (arg == "--abs-tolerance") {
      options.abs_tolerance = ParseFloat(require_value("--abs-tolerance"), "--abs-tolerance");
    } else if (arg == "--rel-tolerance") {
      options.rel_tolerance = ParseFloat(require_value("--rel-tolerance"), "--rel-tolerance");
    } else {
      throw std::runtime_error("Unknown argument: " + std::string(arg));
    }
  }

  if (options.repetitions == 0) {
    throw std::runtime_error("repetitions must be greater than zero.");
  }
  if (options.abs_tolerance < 0.0f || options.rel_tolerance < 0.0f) {
    throw std::runtime_error("Tolerances must be non-negative.");
  }
  return options;
}

std::vector<ComplexValue> LoadBlob(const std::filesystem::path& path, size_t expected_count) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("Failed to open blob: " + path.string());
  }

  const std::streamsize size_bytes = file.tellg();
  const std::streamsize expected_bytes =
      static_cast<std::streamsize>(expected_count * sizeof(ComplexValue));
  if (size_bytes != expected_bytes) {
    throw std::runtime_error("Unexpected blob size for " + path.string() + ". Expected " +
                             std::to_string(expected_bytes) + " bytes, got " +
                             std::to_string(size_bytes) + " bytes.");
  }

  file.seekg(0, std::ios::beg);
  std::vector<ComplexValue> values(expected_count);
  if (!file.read(reinterpret_cast<char*>(values.data()), expected_bytes)) {
    throw std::runtime_error("Failed to read blob: " + path.string());
  }
  return values;
}

bool ComponentWithinTolerance(float actual, float expected, float abs_tolerance,
                              float rel_tolerance) {
  const float diff = std::fabs(actual - expected);
  const float scale = std::max(std::fabs(actual), std::fabs(expected));
  return diff <= abs_tolerance + rel_tolerance * scale;
}

void CompareOutputs(const std::vector<ComplexValue>& actual,
                    const std::vector<ComplexValue>& expected,
                    uint32_t repetition,
                    const Options& options,
                    ComparisonStats& stats,
                    std::vector<std::string>& mismatch_lines) {
  for (size_t index = 0; index < actual.size(); ++index) {
    const ComplexValue& actual_value = actual[index];
    const ComplexValue& expected_value = expected[index];

    const float real_diff = std::fabs(actual_value.real - expected_value.real);
    const float imag_diff = std::fabs(actual_value.imag - expected_value.imag);
    const float abs_diff = std::max(real_diff, imag_diff);
    const float scale = std::max({std::fabs(actual_value.real), std::fabs(actual_value.imag),
                                  std::fabs(expected_value.real), std::fabs(expected_value.imag),
                                  1.0f});
    const float rel_diff = abs_diff / scale;

    if (abs_diff > stats.max_abs_diff) {
      stats.max_abs_diff = abs_diff;
      stats.max_abs_index = index;
    }
    if (rel_diff > stats.max_rel_diff) {
      stats.max_rel_diff = rel_diff;
      stats.max_rel_index = index;
    }

    const bool match =
        ComponentWithinTolerance(actual_value.real, expected_value.real, options.abs_tolerance,
                                 options.rel_tolerance) &&
        ComponentWithinTolerance(actual_value.imag, expected_value.imag, options.abs_tolerance,
                                 options.rel_tolerance);

    if (!match) {
      ++stats.mismatches;
      if (mismatch_lines.size() < options.first_mismatch_limit) {
        const uint32_t batch = static_cast<uint32_t>(index / kFftLength);
        const uint32_t fft_index = static_cast<uint32_t>(index % kFftLength);
        std::ostringstream line;
        line << "Mismatch[" << index << "] batch=" << batch << " element=" << fft_index
             << " repetition=" << repetition << " expected=" << FormatComplex(expected_value)
             << " actual=" << FormatComplex(actual_value)
             << " max_component_diff=" << std::scientific << std::setprecision(8) << abs_diff;
        mismatch_lines.push_back(line.str());
      }
    }
  }
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
    std::cout << "    Scalar block layout: "
              << (device.has_scalar_block_layout ? "present" : "absent") << "\n";
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
  std::cout << "Scalar block layout: "
            << (device.has_scalar_block_layout ? "present" : "absent") << "\n";
  std::cout << "Shader variant: " << VariantName(options.shader_variant) << "\n";
  std::cout << "FFT shape: (" << kFftBatchCount << ", " << kFftLength << ")\n";
  std::cout << "Workgroups: " << kFftBatchCount << "\n";
  std::cout << "Local size X: " << kLocalSizeX << "\n";
  std::cout << "Input blob: " << options.input_path << "\n";
  std::cout << "Reference blob: " << options.reference_path << "\n";
  std::cout << "Host repetitions: " << options.repetitions << "\n";
  std::cout << "Abs tolerance: " << std::scientific << options.abs_tolerance << "\n";
  std::cout << "Rel tolerance: " << std::scientific << options.rel_tolerance << "\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseOptions(argc, argv);

    if (options.list_devices) {
      PrintDeviceList(VulkanContext::EnumerateDevices());
      return 0;
    }

    const std::filesystem::path shader_path =
        VariantShaderPath(options.shader_dir, options.shader_variant);
    if (!std::filesystem::exists(shader_path)) {
      throw std::runtime_error("Shader file not found: " + shader_path.string());
    }

    const std::vector<ComplexValue> input_blob = LoadBlob(options.input_path, kComplexValueCount);
    const std::vector<ComplexValue> reference_blob =
        LoadBlob(options.reference_path, kComplexValueCount);

    VulkanContext context(options.device_index);
    PrintRunHeader(context, options);

    ComputePipeline pipeline(context, shader_path, kComplexValueCount);
    if (options.dump_spirv_info) {
      std::cout << "Shader path: " << shader_path << "\n";
      std::cout << FormatSpirvSummary(pipeline.spirv_summary()) << "\n";
    }

    ComparisonStats stats;
    std::optional<uint32_t> first_failing_repetition;
    std::vector<std::string> mismatch_lines;

    for (uint32_t repetition = 0; repetition < options.repetitions; ++repetition) {
      if (options.verbose) {
        std::cout << "Running repetition " << repetition + 1 << "/" << options.repetitions
                  << "\n";
      }

      const std::vector<ComplexValue> gpu_output =
          pipeline.Run({kFftBatchCount}, input_blob);
      const uint64_t mismatches_before = stats.mismatches;
      CompareOutputs(gpu_output, reference_blob, repetition, options, stats, mismatch_lines);
      if (stats.mismatches > mismatches_before && !first_failing_repetition.has_value()) {
        first_failing_repetition = repetition;
      }
    }

    if (first_failing_repetition.has_value()) {
      std::cout << "First failing repetition: " << *first_failing_repetition << "\n";
      for (const std::string& line : mismatch_lines) {
        std::cout << line << "\n";
      }
    }

    std::cout << "Mismatches: " << stats.mismatches << "\n";
    std::cout << "Max component abs diff: " << std::scientific << stats.max_abs_diff << "\n";
    std::cout << "Max component rel diff: " << std::scientific << stats.max_rel_diff
              << " at index " << stats.max_rel_index << "\n";
    std::cout << "Status: " << (stats.mismatches == 0 ? "PASS" : "FAIL") << "\n";
    return stats.mismatches == 0 ? 0 : 1;
  } catch (const std::exception& exception) {
    std::cerr << "Runtime error: " << exception.what() << "\n";
    return 2;
  }
}
