import Foundation
import Metal

private let fftCount = 875
private let fftLength = 125
private let threadsPerThreadgroup = 25
private let defaultIterations = 10
private let defaultAbsTolerance: Float = 5e-3
private let defaultRelTolerance: Float = 5e-4
private let mismatchPreviewLimit = 5
private let defaultFunctionName = "main0"

private struct Variant {
    let title: String
    let sourcePath: String
    let functionName: String
}

private enum BufferBinding {
    case direct(bufferIndex: Int)
    case argumentBuffer(rootIndex: Int, resourceIndex: Int)
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
    var shaderDumpDir = "../vulkan/shader_dump"
    var functionName = defaultFunctionName
    var iterations = defaultIterations
    var absTolerance = defaultAbsTolerance
    var relTolerance = defaultRelTolerance
}

private enum TestError: Error, CustomStringConvertible {
    case noMetalDevice
    case invalidArguments(String)
    case shaderDumpDirNotFound(String)
    case noDumpedShadersFound(String)
    case shaderSourceReadFailed(path: String, reason: String)
    case shaderCompileFailed(path: String, reason: String)
    case blobNotFound(String)
    case blobSizeMismatch(path: String, expectedBytes: Int, actualBytes: Int)
    case functionNotFound(path: String, name: String)
    case unsupportedShaderLayout(path: String, reason: String)
    case commandBufferFailed(String)
    case commandEncodingFailed(String)

    var description: String {
        switch self {
        case .noMetalDevice:
            return "No Metal device is available on this machine."
        case .invalidArguments(let message):
            return message
        case .shaderDumpDirNotFound(let path):
            return "Shader dump directory not found at \(path). Run the Vulkan repro first or pass --shader-dump-dir."
        case .noDumpedShadersFound(let path):
            return "No dumped Metal shaders were found in \(path). Expected one or more .metal files."
        case .shaderSourceReadFailed(let path, let reason):
            return "Failed to read dumped shader source at \(path): \(reason)"
        case .shaderCompileFailed(let path, let reason):
            return "Failed to compile dumped shader source at \(path): \(reason)"
        case .blobNotFound(let path):
            return "Required blob not found at \(path). Run: python3 generate_blobs.py --output-dir data"
        case .blobSizeMismatch(let path, let expectedBytes, let actualBytes):
            return "Blob size mismatch for \(path). Expected \(expectedBytes) bytes, got \(actualBytes)."
        case .functionNotFound(let path, let name):
            return "Failed to find kernel function '\(name)' in dumped shader \(path)."
        case .unsupportedShaderLayout(let path, let reason):
            return "Unsupported dumped shader layout in \(path): \(reason)"
        case .commandBufferFailed(let reason):
            return "GPU command buffer failed: \(reason)"
        case .commandEncodingFailed(let reason):
            return "Failed to encode GPU work: \(reason)"
        }
    }
}

private func usage() -> String {
    """
    Usage: ./barrier_test.exec [--input PATH] [--reference PATH] [--shader-dump-dir PATH] [--function-name NAME] [--iterations N] [--abs-tol VALUE] [--rel-tol VALUE]

    Defaults:
      --input ../data/fft_875x125_input.bin
      --reference ../data/fft_875x125_reference.bin
      --shader-dump-dir ../vulkan/shader_dump
      --function-name \(defaultFunctionName)
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
        case "--shader-dump-dir":
            guard let value = args.first else {
                throw TestError.invalidArguments("Missing value for --shader-dump-dir.\n\n\(usage())")
            }
            config.shaderDumpDir = value
            args.removeFirst()
        case "--function-name":
            guard let value = args.first, !value.isEmpty else {
                throw TestError.invalidArguments("Invalid value for --function-name.\n\n\(usage())")
            }
            config.functionName = value
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

private func discoverVariants(config: Config) throws -> [Variant] {
    let directoryURL = URL(fileURLWithPath: config.shaderDumpDir, isDirectory: true)
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: directoryURL.path, isDirectory: &isDirectory), isDirectory.boolValue else {
        throw TestError.shaderDumpDirNotFound(directoryURL.path)
    }

    let candidates = try FileManager.default.contentsOfDirectory(
        at: directoryURL,
        includingPropertiesForKeys: [.isRegularFileKey],
        options: [.skipsHiddenFiles]
    )

    let variants = candidates
        .filter { $0.pathExtension == "metal" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
        .map {
            Variant(
                title: $0.lastPathComponent,
                sourcePath: $0.path,
                functionName: config.functionName
            )
        }

    guard !variants.isEmpty else {
        throw TestError.noDumpedShadersFound(directoryURL.path)
    }

    return variants
}

private func makeLibrary(device: MTLDevice, variant: Variant) throws -> MTLLibrary {
    let source: String
    do {
        source = try String(contentsOfFile: variant.sourcePath, encoding: .utf8)
    } catch {
        throw TestError.shaderSourceReadFailed(path: variant.sourcePath, reason: error.localizedDescription)
    }

    do {
        return try device.makeLibrary(source: source, options: nil)
    } catch {
        throw TestError.shaderCompileFailed(path: variant.sourcePath, reason: error.localizedDescription)
    }
}

private func firstMatch(in text: String, pattern: String) -> NSTextCheckingResult? {
    guard let regex = try? NSRegularExpression(pattern: pattern) else {
        return nil
    }
    let range = NSRange(text.startIndex..<text.endIndex, in: text)
    return regex.firstMatch(in: text, options: [], range: range)
}

private func captureInt(in text: String, match: NSTextCheckingResult, group: Int) -> Int? {
    let range = match.range(at: group)
    guard range.location != NSNotFound, let swiftRange = Range(range, in: text) else {
        return nil
    }
    return Int(text[swiftRange])
}

private func detectBufferBinding(source: String, variant: Variant) throws -> BufferBinding {
    let functionName = NSRegularExpression.escapedPattern(for: variant.functionName)
    let directPattern = "kernel\\s+void\\s+\(functionName)\\s*\\([^\\)]*\\bDataBuffer\\s*\\*\\s*\\w+\\s*\\[\\[buffer\\((\\d+)\\)\\]\\]"
    if let match = firstMatch(in: source, pattern: directPattern),
       let bufferIndex = captureInt(in: source, match: match, group: 1) {
        return .direct(bufferIndex: bufferIndex)
    }

    let argumentBufferPattern = "kernel\\s+void\\s+\(functionName)\\s*\\([^\\)]*\\bspvDescriptorSetBuffer\\w*\\s*&\\s*\\w+\\s*\\[\\[buffer\\((\\d+)\\)\\]\\]"
    let resourcePattern = "\\bdevice\\s+DataBuffer\\s*\\*\\s*\\w+\\s*\\[\\[id\\((\\d+)\\)\\]\\]"
    if let rootMatch = firstMatch(in: source, pattern: argumentBufferPattern),
       let rootIndex = captureInt(in: source, match: rootMatch, group: 1),
       let resourceMatch = firstMatch(in: source, pattern: resourcePattern),
       let resourceIndex = captureInt(in: source, match: resourceMatch, group: 1) {
        return .argumentBuffer(rootIndex: rootIndex, resourceIndex: resourceIndex)
    }

    throw TestError.unsupportedShaderLayout(
        path: variant.sourcePath,
        reason: "Could not determine how the FFT IO buffer is bound."
    )
}

private func runVariant(device: MTLDevice,
                        queue: MTLCommandQueue,
                        variant: Variant,
                        inputData: Data,
                        reference: [Complex32],
                        config: Config) throws -> [IterationResult] {
    let source = try String(contentsOfFile: variant.sourcePath, encoding: .utf8)
    let library = try makeLibrary(device: device, variant: variant)
    guard let function = library.makeFunction(name: variant.functionName) else {
        throw TestError.functionNotFound(path: variant.sourcePath, name: variant.functionName)
    }
    let bufferBinding = try detectBufferBinding(source: source, variant: variant)

    let pipeline = try device.makeComputePipelineState(function: function)
    guard let ioBuffer = device.makeBuffer(length: inputData.count, options: .storageModeShared) else {
        throw TestError.commandEncodingFailed("Unable to allocate FFT IO buffer.")
    }
    let argumentBuffer: MTLBuffer?
    switch bufferBinding {
    case .direct:
        argumentBuffer = nil
    case .argumentBuffer(let rootIndex, let resourceIndex):
        let argumentEncoder = function.makeArgumentEncoder(bufferIndex: rootIndex)
        guard let encodedBuffer = device.makeBuffer(length: argumentEncoder.encodedLength, options: .storageModeShared) else {
            throw TestError.commandEncodingFailed("Unable to allocate Metal argument buffer.")
        }
        argumentEncoder.setArgumentBuffer(encodedBuffer, offset: 0)
        argumentEncoder.setBuffer(ioBuffer, offset: 0, index: resourceIndex)
        argumentBuffer = encodedBuffer
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
        switch bufferBinding {
        case .direct(let bufferIndex):
            encoder.setBuffer(ioBuffer, offset: 0, index: bufferIndex)
        case .argumentBuffer(let rootIndex, _):
            guard let argumentBuffer else {
                throw TestError.commandEncodingFailed("Argument buffer was not initialized.")
            }
            encoder.setBuffer(argumentBuffer, offset: 0, index: rootIndex)
            encoder.useResource(ioBuffer, usage: [.read, .write])
        }
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

private func main() throws {
    let config = try parseArgs()
    let expectedBytes = expectedByteCount()

    print("=== Metal Dumped Shader FFT Validation ===")

    guard let device = MTLCreateSystemDefaultDevice() else {
        throw TestError.noMetalDevice
    }

    let variants = try discoverVariants(config: config)

    print("Device: \(device.name)")
    print("macOS: \(ProcessInfo.processInfo.operatingSystemVersionString)")
    print("FFT layout: \(fftCount) x \(fftLength)")
    print("Threads per threadgroup: \(threadsPerThreadgroup)")
    print("Iterations per shader: \(config.iterations)")
    print("Tolerance: abs <= \(config.absTolerance), rel <= \(config.relTolerance)")
    print("Input blob: \(config.inputPath)")
    print("Reference blob: \(config.referencePath)")
    print("Shader dump dir: \(config.shaderDumpDir)")
    print("Kernel function: \(config.functionName)")
    print("Discovered dumped shaders: \(variants.count)")
    print("Expected bytes per blob: \(expectedBytes)")
    print("")

    guard let queue = device.makeCommandQueue() else {
        throw TestError.commandEncodingFailed("Unable to create Metal command queue.")
    }

    let inputData = try loadBlob(at: config.inputPath)
    let referenceData = try loadBlob(at: config.referencePath)
    _ = try decodeComplexBlob(inputData, path: config.inputPath)
    let reference = try decodeComplexBlob(referenceData, path: config.referencePath)

    for (index, variant) in variants.enumerated() {
        print("--- Test \(index + 1): \(variant.title) ---")
        let results = try runVariant(
            device: device,
            queue: queue,
            variant: variant,
            inputData: inputData,
            reference: reference,
            config: config
        )

        for (iterationIndex, result) in results.enumerated() {
            printIterationResult(result, iteration: iterationIndex + 1, totalCount: reference.count)
        }

        printSummary(results: results)
    }
}

do {
    try main()
} catch {
    fputs("ERROR: \(error)\n", stderr)
    exit(1)
}
