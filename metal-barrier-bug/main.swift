import Foundation
import Metal

private let threadgroupSize = 256
private let threadgroupCount = 1024
private let rounds = 8
private let iterations = 10
private let mismatchPreviewLimit = 5

private struct Variant {
    let title: String
    let functionName: String
}

private struct Mismatch {
    let index: Int
    let expected: Float
    let actual: Float
}

private struct IterationResult {
    let mismatchCount: Int
    let mismatches: [Mismatch]
}

private enum TestError: Error, CustomStringConvertible {
    case noMetalDevice
    case libraryNotFound(String)
    case functionNotFound(String)
    case commandBufferFailed(String)
    case commandEncodingFailed(String)

    var description: String {
        switch self {
        case .noMetalDevice:
            return "No Metal device is available on this machine."
        case .libraryNotFound(let path):
            return "Failed to load Metal library at \(path). Build the project first with ./build.sh."
        case .functionNotFound(let name):
            return "Failed to find kernel function '\(name)' in default.metallib."
        case .commandBufferFailed(let reason):
            return "GPU command buffer failed: \(reason)"
        case .commandEncodingFailed(let reason):
            return "Failed to encode GPU work: \(reason)"
        }
    }
}

private func makeValue(localID: Int, round: Int) -> Float {
    Float(localID * 37 + round * 1000 + 7)
}

private func makeReference() -> [Float] {
    let totalThreads = threadgroupSize * threadgroupCount
    var reference = Array(repeating: Float.zero, count: totalThreads)

    for group in 0..<threadgroupCount {
        let baseIndex = group * threadgroupSize
        for localID in 0..<threadgroupSize {
            var accumulator: Float = 0
            for round in 0..<rounds {
                let sourceLocalID = (localID + round + 1) & (threadgroupSize - 1)
                accumulator += makeValue(localID: sourceLocalID, round: round)
            }
            reference[baseIndex + localID] = accumulator
        }
    }

    return reference
}

private func evaluateResults(actual: UnsafeBufferPointer<Float>, reference: [Float]) -> IterationResult {
    var mismatchCount = 0
    var mismatches: [Mismatch] = []

    for index in 0..<reference.count {
        let actualValue = actual[index]
        let expectedValue = reference[index]
        if actualValue != expectedValue {
            mismatchCount += 1
            if mismatches.count < mismatchPreviewLimit {
                mismatches.append(Mismatch(index: index, expected: expectedValue, actual: actualValue))
            }
        }
    }

    return IterationResult(mismatchCount: mismatchCount, mismatches: mismatches)
}

private func runVariant(device: MTLDevice,
                        queue: MTLCommandQueue,
                        library: MTLLibrary,
                        variant: Variant,
                        reference: [Float]) throws -> [IterationResult] {
    guard let function = library.makeFunction(name: variant.functionName) else {
        throw TestError.functionNotFound(variant.functionName)
    }

    let pipeline = try device.makeComputePipelineState(function: function)
    let totalThreads = threadgroupSize * threadgroupCount
    let bufferLength = totalThreads * MemoryLayout<Float>.stride
    guard let outputBuffer = device.makeBuffer(length: bufferLength, options: .storageModeShared) else {
        throw TestError.commandEncodingFailed("Unable to allocate output buffer.")
    }

    let threadsPerThreadgroup = MTLSize(width: threadgroupSize, height: 1, depth: 1)
    let threadgroupsPerGrid = MTLSize(width: threadgroupCount, height: 1, depth: 1)

    var results: [IterationResult] = []
    results.reserveCapacity(iterations)

    for _ in 0..<iterations {
        memset(outputBuffer.contents(), 0, bufferLength)

        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw TestError.commandEncodingFailed("Unable to create command buffer or compute encoder.")
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(outputBuffer, offset: 0, index: 0)
        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if commandBuffer.status == .error {
            let reason = commandBuffer.error?.localizedDescription ?? "unknown command buffer error"
            throw TestError.commandBufferFailed(reason)
        }

        let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: totalThreads)
        let actual = UnsafeBufferPointer(start: pointer, count: totalThreads)
        results.append(evaluateResults(actual: actual, reference: reference))
    }

    return results
}

private func printIterationResult(_ result: IterationResult, iteration: Int, totalCount: Int) {
    let status = result.mismatchCount == 0 ? "PASS" : "FAIL"
    print("  Iteration \(iteration): \(result.mismatchCount) mismatches out of \(totalCount) (\(status))")

    for mismatch in result.mismatches {
        let group = mismatch.index / threadgroupSize
        let localID = mismatch.index % threadgroupSize
        print("    group \(group), local \(localID), index \(mismatch.index): expected \(mismatch.expected), got \(mismatch.actual)")
    }
}

private func printSummary(for variant: Variant, results: [IterationResult]) {
    let failingIterations = results.filter { $0.mismatchCount > 0 }.count
    if failingIterations == 0 {
        print("  Result: PASS (0/\(results.count) iterations had mismatches)")
    } else {
        print("  Result: FAIL (mismatches in \(failingIterations)/\(results.count) iterations)")
    }
    print("")
}

private func printConclusion(threadgroupOnlyFailures: Int, allFlagsFailures: Int) {
    print("=== CONCLUSION ===")

    if threadgroupOnlyFailures > 0 && allFlagsFailures == 0 {
        print("threadgroup_barrier with mem_threadgroup alone did NOT reliably")
        print("synchronize threadgroup memory on this machine.")
        print("Adding mem_device | mem_threadgroup | mem_texture worked around the issue.")
    } else if threadgroupOnlyFailures == 0 && allFlagsFailures == 0 {
        print("This machine did not reproduce the bug in either variant.")
        print("That is still useful signal and may indicate the issue is hardware- or OS-specific.")
    } else if threadgroupOnlyFailures > 0 && allFlagsFailures > 0 {
        print("Both variants produced incorrect results on this machine.")
        print("That suggests either a broader issue or that this stress test needs refinement here.")
    } else {
        print("The broader-flags variant failed while the mem_threadgroup-only variant passed.")
        print("That does not match the expected pattern and should be investigated separately.")
    }
}

private func main() throws {
    print("=== Metal threadgroup_barrier Bug Reproduction ===")

    guard let device = MTLCreateSystemDefaultDevice() else {
        throw TestError.noMetalDevice
    }

    print("Device: \(device.name)")
    print("macOS: \(ProcessInfo.processInfo.operatingSystemVersionString)")
    print("Threadgroups: \(threadgroupCount)")
    print("Threads per threadgroup: \(threadgroupSize)")
    print("Rounds per dispatch: \(rounds)")
    print("Iterations per variant: \(iterations)")
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
    let reference = makeReference()
    let totalCount = reference.count

    let variants = [
        Variant(
            title: "threadgroup_barrier(mem_flags::mem_threadgroup)",
            functionName: "test_barrier_threadgroup_only"
        ),
        Variant(
            title: "threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture)",
            functionName: "test_barrier_all_flags"
        ),
    ]

    var failingIterationCounts: [String: Int] = [:]

    for (index, variant) in variants.enumerated() {
        print("--- Test \(index + 1): \(variant.title) ---")
        let results = try runVariant(device: device, queue: queue, library: library, variant: variant, reference: reference)
        for (iterationIndex, result) in results.enumerated() {
            printIterationResult(result, iteration: iterationIndex + 1, totalCount: totalCount)
        }
        printSummary(for: variant, results: results)
        failingIterationCounts[variant.functionName] = results.filter { $0.mismatchCount > 0 }.count
    }

    let threadgroupOnlyFailures = failingIterationCounts["test_barrier_threadgroup_only"] ?? 0
    let allFlagsFailures = failingIterationCounts["test_barrier_all_flags"] ?? 0
    printConclusion(threadgroupOnlyFailures: threadgroupOnlyFailures, allFlagsFailures: allFlagsFailures)
}

do {
    try main()
} catch {
    fputs("ERROR: \(error)\n", stderr)
    exit(1)
}
