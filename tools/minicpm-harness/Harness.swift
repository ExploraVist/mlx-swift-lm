//
//  minicpm-harness — parity check for the Swift MiniCPM-V-4.6 port.
//
//  Usage:
//    swift run minicpm-harness <model-dir> <image-path> <prompt...>
//
//  Loads the already-downloaded HF snapshot from <model-dir>, runs greedy
//  (temperature 0) generation on the image + prompt, and prints the result
//  for comparison against Python mlx_vlm ground truth.
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
            print("usage: minicpm-harness <model-dir|--remote> <image-path> [prompt...]")
            exit(2)
        }
        let imagePath = URL(fileURLWithPath: args[2])
        let prompt = args.count > 3 ? args[3...].joined(separator: " ") : "Describe this image."

        MLX.GPU.set(cacheLimit: 1024 * 1024 * 1024)

        let container: ModelContainer
        if args[1] == "--remote" {
            // Exercise the exact remote path the iOS app uses.
            print("Remote download via #hubDownloader() …")
            let box = ProgressBox()
            container = try await VLMModelFactory.shared.loadContainer(
                from: #hubDownloader(),
                using: #huggingFaceTokenizerLoader(),
                configuration: VLMRegistry.minicpmV46_4bit
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
        print("Model loaded.")

        let result = try await container.perform { (context: ModelContext) -> String in
            var userInput = UserInput(
                prompt: .text(prompt),
                images: [.url(imagePath)]
            )
            // No pre-resize: the MiniCPM processor does its own slicing math,
            // and parity with Python requires the original resolution.
            userInput.processing = .init()

            let lmInput = try await context.processor.prepare(input: userInput)
            print("Prompt tokens: \(lmInput.text.tokens.size)")

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
                            + String(format: "%.1f tok/s", info.tokensPerSecond))
                default:
                    break
                }
            }
            return output
        }

        print("\n=== DONE ===")
        print(result)
    }
}
