import Foundation
import Metal

private let fftCount = 875
private let fftLength = 125
private let threadsPerThreadgroup = 25
private let defaultIterations = 10
private let defaultAbsTolerance: Float = 5e-3
private let defaultRelTolerance: Float = 5e-4
private let mismatchPreviewLimit = 5

private struct Variant {
    let title: String
    let functionName: String
}

private struct Complex32 {
    var real: Float
    var imag: Float
}

private struct Mismatch {
    let index: Int
    let expected: Complex32
    let actual: Complex32
    let absError: Float
}

private struct IterationResult {
    let mismatchCount: Int
    let mismatches: [Mismatch]
    let maxAbsError: Float
}

private struct Config {
    var inputPath = "../data/fft_875x125_input.bin"
    var referencePath = "../data/fft_875x125_reference.bin"
    var iterations = defaultIterations
    var absTolerance = defaultAbsTolerance
    var relTolerance = defaultRelTolerance
}

private enum TestError: Error, CustomStringConvertible {
    case noMetalDevice
    case invalidArguments(String)
    case libraryNotFound(String)
    case blobNotFound(String)
    case blobSizeMismatch(path: String, expectedBytes: Int, actualBytes: Int)
    case functionNotFound(String)
    case commandBufferFailed(String)
    case commandEncodingFailed(String)

    var description: String {
        switch self {
        case .noMetalDevice:
            return "No Metal device is available on this machine."
        case .invalidArguments(let message):
            return message
        case .libraryNotFound(let path):
            return "Failed to load Metal library at \(path). Build the project first with ./build.sh."
        case .blobNotFound(let path):
            return "Required blob not found at \(path). Run: python3 generate_blobs.py --output-dir data"
        case .blobSizeMismatch(let path, let expectedBytes, let actualBytes):
            return "Blob size mismatch for \(path). Expected \(expectedBytes) bytes, got \(actualBytes)."
        case .functionNotFound(let name):
            return "Failed to find kernel function '\(name)' in default.metallib."
        case .commandBufferFailed(let reason):
            return "GPU command buffer failed: \(reason)"
        case .commandEncodingFailed(let reason):
            return "Failed to encode GPU work: \(reason)"
        }
    }
}

private func usage() -> String {
    """
    Usage: ./barrier_test.exec [--input PATH] [--reference PATH] [--iterations N] [--abs-tol VALUE] [--rel-tol VALUE]

    Defaults:
      --input ../data/fft_875x125_input.bin
      --reference ../data/fft_875x125_reference.bin
      --iterations \(defaultIterations)
      --abs-tol \(defaultAbsTolerance)
      --rel-tol \(defaultRelTolerance)
    """
}

private func parseArgs() throws -> Config {
    var config = Config()
    var args = Array(CommandLine.arguments.dropFirst())

    while !args.isEmpty {
        let flag = args.removeFirst()
        switch flag {
        case "--help", "-h":
            print(usage())
            exit(0)
        case "--input":
            guard let value = args.first else {
                throw TestError.invalidArguments("Missing value for --input.\n\n\(usage())")
            }
            config.inputPath = value
            args.removeFirst()
        case "--reference":
            guard let value = args.first else {
                throw TestError.invalidArguments("Missing value for --reference.\n\n\(usage())")
            }
            config.referencePath = value
            args.removeFirst()
        case "--iterations":
            guard let value = args.first, let iterations = Int(value), iterations > 0 else {
                throw TestError.invalidArguments("Invalid value for --iterations.\n\n\(usage())")
            }
            config.iterations = iterations
            args.removeFirst()
        case "--abs-tol":
            guard let value = args.first, let tolerance = Float(value), tolerance >= 0 else {
                throw TestError.invalidArguments("Invalid value for --abs-tol.\n\n\(usage())")
            }
            config.absTolerance = tolerance
            args.removeFirst()
        case "--rel-tol":
            guard let value = args.first, let tolerance = Float(value), tolerance >= 0 else {
                throw TestError.invalidArguments("Invalid value for --rel-tol.\n\n\(usage())")
            }
            config.relTolerance = tolerance
            args.removeFirst()
        default:
            throw TestError.invalidArguments("Unknown argument: \(flag)\n\n\(usage())")
        }
    }

    return config
}

private func loadBlob(at path: String) throws -> Data {
    guard FileManager.default.fileExists(atPath: path) else {
        throw TestError.blobNotFound(path)
    }
    return try Data(contentsOf: URL(fileURLWithPath: path))
}

private func expectedByteCount() -> Int {
    fftCount * fftLength * MemoryLayout<Complex32>.stride
}

private func decodeComplexBlob(_ data: Data, path: String) throws -> [Complex32] {
    let expectedBytes = expectedByteCount()
    guard data.count == expectedBytes else {
        throw TestError.blobSizeMismatch(path: path, expectedBytes: expectedBytes, actualBytes: data.count)
    }

    return data.withUnsafeBytes { rawBuffer in
        let typed = rawBuffer.bindMemory(to: Complex32.self)
        return Array(typed)
    }
}

private func toleranceLimit(expected: Complex32, absTolerance: Float, relTolerance: Float) -> Float {
    let magnitude = hypotf(expected.real, expected.imag)
    return max(absTolerance, magnitude * relTolerance)
}

private func evaluateResults(actual: UnsafeBufferPointer<Complex32>,
                             reference: [Complex32],
                             absTolerance: Float,
                             relTolerance: Float) -> IterationResult {
    var mismatchCount = 0
    var mismatches: [Mismatch] = []
    var maxAbsError: Float = 0

    for index in 0..<reference.count {
        let actualValue = actual[index]
        let expectedValue = reference[index]
        let dx = actualValue.real - expectedValue.real
        let dy = actualValue.imag - expectedValue.imag
        let absError = hypotf(dx, dy)
        maxAbsError = max(maxAbsError, absError)

        if absError > toleranceLimit(expected: expectedValue, absTolerance: absTolerance, relTolerance: relTolerance) {
            mismatchCount += 1
            if mismatches.count < mismatchPreviewLimit {
                mismatches.append(
                    Mismatch(
                        index: index,
                        expected: expectedValue,
                        actual: actualValue,
                        absError: absError
                    )
                )
            }
        }
    }

    return IterationResult(mismatchCount: mismatchCount, mismatches: mismatches, maxAbsError: maxAbsError)
}

private func refillBuffer(_ buffer: MTLBuffer, from data: Data) {
    _ = data.withUnsafeBytes { rawBuffer in
        memcpy(buffer.contents(), rawBuffer.baseAddress!, rawBuffer.count)
    }
}

private func runVariant(device: MTLDevice,
                        queue: MTLCommandQueue,
                        library: MTLLibrary,
                        variant: Variant,
                        inputData: Data,
                        reference: [Complex32],
                        config: Config) throws -> [IterationResult] {
    guard let function = library.makeFunction(name: variant.functionName) else {
        throw TestError.functionNotFound(variant.functionName)
    }

    let pipeline = try device.makeComputePipelineState(function: function)
    guard let ioBuffer = device.makeBuffer(length: inputData.count, options: .storageModeShared) else {
        throw TestError.commandEncodingFailed("Unable to allocate FFT IO buffer.")
    }

    let threadgroupsPerGrid = MTLSize(width: fftCount, height: 1, depth: 1)
    let threads = MTLSize(width: threadsPerThreadgroup, height: 1, depth: 1)

    var results: [IterationResult] = []
    results.reserveCapacity(config.iterations)

    for _ in 0..<config.iterations {
        refillBuffer(ioBuffer, from: inputData)

        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw TestError.commandEncodingFailed("Unable to create command buffer or compute encoder.")
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(ioBuffer, offset: 0, index: 0)
        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threads)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if commandBuffer.status == .error {
            let reason = commandBuffer.error?.localizedDescription ?? "unknown command buffer error"
            throw TestError.commandBufferFailed(reason)
        }

        let pointer = ioBuffer.contents().bindMemory(to: Complex32.self, capacity: reference.count)
        let actual = UnsafeBufferPointer(start: pointer, count: reference.count)
        results.append(
            evaluateResults(
                actual: actual,
                reference: reference,
                absTolerance: config.absTolerance,
                relTolerance: config.relTolerance
            )
        )
    }

    return results
}

private func printIterationResult(_ result: IterationResult, iteration: Int, totalCount: Int) {
    let status = result.mismatchCount == 0 ? "PASS" : "FAIL"
    print("  Iteration \(iteration): \(result.mismatchCount) mismatches out of \(totalCount) (\(status)), max abs error \(result.maxAbsError)")

    for mismatch in result.mismatches {
        let fftIndex = mismatch.index / fftLength
        let binIndex = mismatch.index % fftLength
        print(
            "    fft \(fftIndex), bin \(binIndex), index \(mismatch.index): " +
            "expected (\(mismatch.expected.real), \(mismatch.expected.imag)), " +
            "got (\(mismatch.actual.real), \(mismatch.actual.imag)), " +
            "abs error \(mismatch.absError)"
        )
    }
}

private func printSummary(results: [IterationResult]) {
    let failingIterations = results.filter { $0.mismatchCount > 0 }.count
    if failingIterations == 0 {
        print("  Result: PASS (0/\(results.count) iterations had mismatches)")
    } else {
        print("  Result: FAIL (mismatches in \(failingIterations)/\(results.count) iterations)")
    }
    print("")
}

private func printConclusion(resultsByFunction: [String: [IterationResult]]) {
    let threadgroupOnlyFailures = resultsByFunction["fft_875x125_threadgroup_only"]?.filter { $0.mismatchCount > 0 }.count ?? 0
    let deviceThreadgroupFailures = resultsByFunction["fft_875x125_device_and_threadgroup"]?.filter { $0.mismatchCount > 0 }.count ?? 0
    let allFlagsFailures = resultsByFunction["fft_875x125_all_flags"]?.filter { $0.mismatchCount > 0 }.count ?? 0

    print("=== CONCLUSION ===")

    if threadgroupOnlyFailures > 0 && deviceThreadgroupFailures == 0 && allFlagsFailures == 0 {
        print("The FFT workload reproduces a synchronization failure with mem_threadgroup alone.")
        print("Adding mem_device is sufficient to make the kernel match the NumPy reference on this machine.")
    } else if threadgroupOnlyFailures == 0 && deviceThreadgroupFailures == 0 && allFlagsFailures == 0 {
        print("This machine did not reproduce the barrier bug with the FFT workload.")
        print("That is still useful data and may indicate hardware- or OS-specific behavior.")
    } else {
        print("Observed failure counts:")
        print("  mem_threadgroup only: \(threadgroupOnlyFailures)")
        print("  mem_device | mem_threadgroup: \(deviceThreadgroupFailures)")
        print("  all flags: \(allFlagsFailures)")
    }
}

private func main() throws {
    let config = try parseArgs()
    let expectedBytes = expectedByteCount()

    print("=== Metal threadgroup_barrier FFT Bug Reproduction ===")

    guard let device = MTLCreateSystemDefaultDevice() else {
        throw TestError.noMetalDevice
    }

    print("Device: \(device.name)")
    print("macOS: \(ProcessInfo.processInfo.operatingSystemVersionString)")
    print("FFT layout: \(fftCount) x \(fftLength)")
    print("Threads per threadgroup: \(threadsPerThreadgroup)")
    print("Iterations per variant: \(config.iterations)")
    print("Tolerance: abs <= \(config.absTolerance), rel <= \(config.relTolerance)")
    print("Input blob: \(config.inputPath)")
    print("Reference blob: \(config.referencePath)")
    print("Expected bytes per blob: \(expectedBytes)")
    print("")

    guard let queue = device.makeCommandQueue() else {
        throw TestError.commandEncodingFailed("Unable to create Metal command queue.")
    }

    let currentDirectory = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let libraryURL = currentDirectory.appendingPathComponent("default.metallib")
    guard FileManager.default.fileExists(atPath: libraryURL.path) else {
        throw TestError.libraryNotFound(libraryURL.path)
    }

    let library = try device.makeLibrary(URL: libraryURL)
    let inputData = try loadBlob(at: config.inputPath)
    let referenceData = try loadBlob(at: config.referencePath)
    _ = try decodeComplexBlob(inputData, path: config.inputPath)
    let reference = try decodeComplexBlob(referenceData, path: config.referencePath)

    let variants = [
        Variant(
            title: "threadgroup_barrier(mem_flags::mem_threadgroup)",
            functionName: "fft_875x125_threadgroup_only"
        ),
        Variant(
            title: "threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup)",
            functionName: "fft_875x125_device_and_threadgroup"
        ),
        Variant(
            title: "threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture)",
            functionName: "fft_875x125_all_flags"
        ),
    ]

    var resultsByFunction: [String: [IterationResult]] = [:]

    for (index, variant) in variants.enumerated() {
        print("--- Test \(index + 1): \(variant.title) ---")
        let results = try runVariant(
            device: device,
            queue: queue,
            library: library,
            variant: variant,
            inputData: inputData,
            reference: reference,
            config: config
        )

        for (iterationIndex, result) in results.enumerated() {
            printIterationResult(result, iteration: iterationIndex + 1, totalCount: reference.count)
        }

        printSummary(results: results)
        resultsByFunction[variant.functionName] = results
    }

    printConclusion(resultsByFunction: resultsByFunction)
}

do {
    try main()
} catch {
    fputs("ERROR: \(error)\n", stderr)
    exit(1)
}
