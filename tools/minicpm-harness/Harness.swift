//
//  minicpm-harness — parity / smoke check for the Swift MiniCPM-V 4.6 and 4.7 ports.
//
//  Usage:
//    swift run minicpm-harness <model-dir|--remote|--remote-v47> <image-path>[,<image-path>...] [prompt...]
//
//  Loads the already-downloaded HF snapshot from <model-dir> (4.6 or 4.7, picked by
//  config.json `model_type`), runs greedy (temperature 0) generation on each image +
//  prompt, and prints the result plus per-image latency and peak MLX memory, for
//  comparison against Python mlx_vlm ground truth. MINICPM_HARNESS_SHOW_PROMPT=1 also
//  prints the decoded prompt.
//

import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
import Tokenizers

final class ProgressBox: @unchecked Sendable {
    private let lock = NSLock()
    private var last = -1
    func changed(_ pct: Int) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if pct == last { return false }
        last = pct
        return true
    }
}

@main
struct Harness {
    static func main() async {
        do {
            try await run()
        } catch {
            print("HARNESS ERROR: \(error)")
            exit(1)
        }
    }

    static func run() async throws {
        let args = CommandLine.arguments
        guard args.count >= 3 else {
            print(
                "usage: minicpm-harness <model-dir|--remote|--remote-v47> "
                    + "<image-path>[,<image-path>...] [prompt...]")
            exit(2)
        }
        let imagePaths = args[2].split(separator: ",").map { URL(fileURLWithPath: String($0)) }
        let showPrompt = ProcessInfo.processInfo.environment["MINICPM_HARNESS_SHOW_PROMPT"] == "1"
        let prompt = args.count > 3 ? args[3...].joined(separator: " ") : "Describe this image."

        MLX.GPU.set(cacheLimit: 1024 * 1024 * 1024)

        let loadStart = Date()
        let container: ModelContainer
        if args[1] == "--remote" || args[1] == "--remote-v47" {
            // Exercise the exact remote path the iOS app uses.
            print("Remote download via #hubDownloader() …")
            let box = ProgressBox()
            container = try await VLMModelFactory.shared.loadContainer(
                from: #hubDownloader(),
                using: #huggingFaceTokenizerLoader(),
                configuration: args[1] == "--remote-v47"
                    ? VLMRegistry.minicpmV47_1B_4bit : VLMRegistry.minicpmV46_4bit
            ) { progress in
                let pct = Int(progress.fractionCompleted * 100)
                if box.changed(pct) {
                    print(
                        "progress: \(pct)% (\(progress.completedUnitCount)/\(progress.totalUnitCount))"
                    )
                }
            }
        } else {
            let modelDir = URL(fileURLWithPath: args[1])
            print("Loading model from \(modelDir.path) …")
            container = try await VLMModelFactory.shared.loadContainer(
                from: modelDir,
                using: #huggingFaceTokenizerLoader()
            )
        }
        print(
            String(
                format: "Model loaded in %.1f s (active %.0f MB).",
                Date().timeIntervalSince(loadStart),
                Double(Memory.activeMemory) / 1e6))

        for imagePath in imagePaths {
            Memory.peakMemory = 0  // reset
            let start = Date()
            let result = try await container.perform { (context: ModelContext) -> String in
                var userInput = UserInput(
                    prompt: .text(prompt),
                    images: [.url(imagePath)]
                )
                // No pre-resize: the MiniCPM processor does its own slicing math,
                // and parity with Python requires the original resolution.
                userInput.processing = .init()

                let lmInput = try await context.processor.prepare(input: userInput)
                print("\n### \(imagePath.lastPathComponent)")
                print("Prompt tokens: \(lmInput.text.tokens.size)")
                if showPrompt {
                    let ids = lmInput.text.tokens.reshaped(-1).asArray(Int32.self).map(Int.init)
                    print("Prompt: \(context.tokenizer.decode(tokenIds: ids).debugDescription)")
                }

                let parameters = GenerateParameters(maxTokens: 120, temperature: 0.0)

                var output = ""
                let stream = try MLXLMCommon.generate(
                    input: lmInput, parameters: parameters, context: context)
                for await item in stream {
                    switch item {
                    case .chunk(let chunk):
                        output += chunk
                        print(chunk, terminator: "")
                        fflush(stdout)
                    case .info(let info):
                        print(
                            "\n---\n\(info.promptTokenCount) prompt tokens, "
                                + "\(info.generationTokenCount) generated, "
                                + String(
                                    format: "prefill %.2f s, %.1f tok/s, stop: %@",
                                    info.promptTime, info.tokensPerSecond,
                                    "\(info.stopReason)"))
                    default:
                        break
                    }
                }
                return output
            }
            print(
                String(
                    format: "=== %@: %.2f s total, peak MLX memory %.0f MB ===",
                    imagePath.lastPathComponent, Date().timeIntervalSince(start),
                    Double(Memory.peakMemory) / 1e6))
            print(result)
        }
        print("\n=== DONE ===")
    }
}
