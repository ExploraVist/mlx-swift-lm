import CoreImage
import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXVLM

/// MiniCPM-V 4.7 (`minicpmv4_7`) support next to 4.6: canvas M-RoPE positions,
/// config decoding, processor-config loading and the 4.7 prompt layout.
final class MiniCPMV47Tests: XCTestCase {

    private let pad = 248_056
    private let imageStart = 248_078
    private let imageEnd = 248_079
    private let sliceStart = 248_088
    private let sliceEnd = 248_089
    private let imageIdStart = 248_090
    private let imageIdEnd = 248_091
    private let newline = 198

    // MARK: - Canvas M-RoPE

    /// Golden vector from the reference `CanvasRoPE.get_rope_index` (mlx-vlm
    /// `minicpmv4_7/rope.py`, identical to transformers `MiniCPMV4_7Model.get_rope_index`):
    /// 3 text tokens, `<image_id>0</image_id>`, a 16x24-patch thumbnail, a 2x2 grid of
    /// 16x16-patch slices, then `\n` + 2 text tokens, 16x downsample.
    func testCanvasPositionsMatchReferenceGoldenVector() throws {
        var ids = [1, 2, 3, imageIdStart, 15, imageIdEnd]
        ids += [imageStart] + Array(repeating: pad, count: 24) + [imageEnd]
        let row = [sliceStart] + Array(repeating: pad, count: 16) + [sliceEnd]
        ids += row + row + [newline] + row + row
        ids += [newline, 7, 8]
        XCTAssertEqual(ids.count, 108)

        let (positions, delta) = try MiniCPMVCanvasRoPE.positionIds(
            inputIds: ids,
            grids: [(16, 24), (16, 16), (16, 16), (16, 16), (16, 16)],
            markers: .minicpmV47,
            mergeFactor: 4)

        let expectedT = csv(
            "0,1,2,3,4,5,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,15,16"
        )
        let expectedH = csv(
            "0,1,2,3,4,5,5,6,6,6,6,6,6,8,8,8,8,8,8,11,11,11,11,11,11,13,13,13,13,13,13,14,6,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9,9,6,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9,9,9,10,10,10,10,10,11,11,11,11,12,12,12,12,13,13,13,13,13,10,10,10,10,10,11,11,11,11,12,12,12,12,13,13,13,13,13,6,15,16"
        )
        let expectedW = csv(
            "0,1,2,3,4,5,5,6,7,9,10,12,13,6,7,9,10,12,13,6,7,9,10,12,13,6,7,9,10,12,13,14,6,6,7,8,9,6,7,8,9,6,7,8,9,6,7,8,9,9,10,10,11,12,13,10,11,12,13,10,11,12,13,10,11,12,13,13,14,6,6,7,8,9,6,7,8,9,6,7,8,9,6,7,8,9,9,10,10,11,12,13,10,11,12,13,10,11,12,13,10,11,12,13,13,6,15,16"
        )

        XCTAssertEqual(delta, -91)
        XCTAssertEqual(positions.count, 3)
        XCTAssertEqual(positions[0], expectedT)
        XCTAssertEqual(positions[1], expectedH)
        XCTAssertEqual(positions[2], expectedW)

        // Spot checks called out in the porting spec.
        func at(_ i: Int) -> [Int] { [positions[0][i], positions[1][i], positions[2][i]] }
        XCTAssertEqual(at(6), [6, 5, 5])  // <image>
        XCTAssertEqual((7 ..< 13).map { positions[2][$0] }, [6, 7, 9, 10, 12, 13])  // thumb row 0 W
        XCTAssertEqual(at(31), [6, 14, 14])  // </image>
        XCTAssertEqual(at(50), [6, 6, 10])  // <slice> of slice 1
        XCTAssertEqual(at(67), [6, 9, 13])  // </slice> of slice 1
        XCTAssertEqual(at(68), [6, 9, 14])  // row separator \n
        XCTAssertEqual(at(105), [6, 6, 6])  // \n after the last slice row
        XCTAssertEqual(at(106), [15, 15, 15])  // next text token
    }

    /// Thumbnail only (no slices), 12x16 patches: the `\n` that follows `</image>` belongs
    /// to the frame and text resumes after the canvas. Reference output from the same script.
    func testCanvasPositionsThumbnailOnly() throws {
        var ids = [5, 6, imageIdStart, 15, imageIdEnd]
        ids += [imageStart] + Array(repeating: pad, count: 12) + [imageEnd]
        ids += [newline, 9, 9, 9]

        let (positions, delta) = try MiniCPMVCanvasRoPE.positionIds(
            inputIds: ids, grids: [(12, 16)], markers: .minicpmV47, mergeFactor: 4)

        XCTAssertEqual(delta, -10)
        XCTAssertEqual(
            positions[0], csv("0,1,2,3,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,10,11,12"))
        XCTAssertEqual(
            positions[1], csv("0,1,2,3,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,5,10,11,12"))
        XCTAssertEqual(
            positions[2], csv("0,1,2,3,4,4,5,6,7,8,5,6,7,8,5,6,7,8,9,5,10,11,12"))
    }

    func testCanvasPositionsTextOnlyIsArange() throws {
        let ids = [248_045, 872, 198, 9_906, 248_046, 198, 248_045, 74_455, 198]
        let (positions, delta) = try MiniCPMVCanvasRoPE.positionIds(
            inputIds: ids, grids: [], markers: .minicpmV47, mergeFactor: 4)
        XCTAssertEqual(delta, 0)
        for axis in positions {
            XCTAssertEqual(axis, Array(0 ..< ids.count))
        }
    }

    func testCanvasPositionsRejectGridMismatch() {
        let ids = [1, imageStart] + Array(repeating: pad, count: 4) + [imageEnd, 2]
        // 4 placeholders but the grid says 16x24 patches -> 24 tokens.
        XCTAssertThrowsError(
            try MiniCPMVCanvasRoPE.positionIds(
                inputIds: ids, grids: [(16, 24)], markers: .minicpmV47, mergeFactor: 4))
        // A visual span with no grid at all.
        XCTAssertThrowsError(
            try MiniCPMVCanvasRoPE.positionIds(
                inputIds: ids, grids: [], markers: .minicpmV47, mergeFactor: 4))
    }

    // MARK: - Config decoding

    func testDecodesV46Configs() throws {
        for json in [MiniCPMVFixtures.openbmbV46Config, MiniCPMVFixtures.mlxV46Config] {
            let config = try JSONDecoder.json5().decode(
                MiniCPMV46Configuration.self, from: Data(json.utf8))
            XCTAssertEqual(config.modelType, "minicpmv4_6")
            XCTAssertFalse(config.usesCanvasRoPE)
            XCTAssertNil(config.mropeMode)
            XCTAssertEqual(config.downsampleMode, "16x")
            XCTAssertEqual(config.imageTokenId, 248_056)
            XCTAssertEqual(config.insertLayerId, 6)
            XCTAssertTrue(config.textConfiguration.tieWordEmbeddings)
            XCTAssertEqual(config.textConfiguration.hiddenSize, 1024)
            XCTAssertEqual(config.visionConfiguration.hiddenLayers, 27)
        }
    }

    func testDecodesV47Config() throws {
        let config = try JSONDecoder.json5().decode(
            MiniCPMV46Configuration.self, from: Data(MiniCPMVFixtures.mlxV47Config.utf8))
        XCTAssertEqual(config.modelType, "minicpmv4_7")
        XCTAssertEqual(config.mropeMode, "canvas")
        XCTAssertTrue(config.usesCanvasRoPE)
        XCTAssertEqual(config.downsampleMode, "16x")
        XCTAssertEqual(config.canvasMergeFactor, 4)
        XCTAssertEqual(config.imageTokenId, 248_056)
        XCTAssertEqual(config.insertLayerId, 6)
        XCTAssertFalse(config.dropVisionLastLayer)
        XCTAssertFalse(config.textConfiguration.tieWordEmbeddings)
        XCTAssertEqual(config.textConfiguration.vocabularySize, 248_094)
        XCTAssertEqual(
            config.textConfiguration.ropeParameters?["mrope_section"]?.asInts(), [11, 11, 10])

        let markers = config.canvasMarkers
        XCTAssertEqual(markers.imageTokenId, 248_056)
        XCTAssertEqual(markers.imageStartId, 248_078)
        XCTAssertEqual(markers.imageEndId, 248_079)
        XCTAssertEqual(markers.sliceStartId, 248_088)
        XCTAssertEqual(markers.sliceEndId, 248_089)
        XCTAssertEqual(markers.newlineId, 198)
    }

    func testModelRegistryBuildsV47WithUntiedHead() async throws {
        let v47 = try await VLMTypeRegistry.shared.createModel(
            configuration: Data(MiniCPMVFixtures.mlxV47Config.utf8), modelType: "minicpmv4_7")
        XCTAssertTrue(v47 is MiniCPMV46)
        let v47Keys = Set(v47.parameters().flattened().map { $0.0 })
        XCTAssertTrue(v47Keys.contains("language_model.lm_head.weight"))

        let v46 = try await VLMTypeRegistry.shared.createModel(
            configuration: Data(MiniCPMVFixtures.mlxV46Config.utf8), modelType: "minicpmv4_6")
        let v46Keys = Set(v46.parameters().flattened().map { $0.0 })
        XCTAssertFalse(v46Keys.contains("language_model.lm_head.weight"))
        // Apart from the untied head, 4.7 has exactly the 4.6 module tree.
        XCTAssertEqual(v47Keys.subtracting(v46Keys), ["language_model.lm_head.weight"])
        XCTAssertEqual(v46Keys.subtracting(v47Keys), [])
    }

    func testRegistryHasV47Configuration() {
        let configuration = VLMRegistry.minicpmV47_1B_4bit
        XCTAssertEqual(configuration.name, "mlx-community/MiniCPM-V-4.7-1B-4bit")
        XCTAssertEqual(configuration.extraEOSTokens, ["<|im_end|>"])
        XCTAssertTrue(VLMRegistry.all().contains { $0.name == configuration.name })
    }

    // MARK: - Processor config loading

    /// mlx-community's 4.7 `preprocessor_config.json` has no `processor_class`; the class
    /// name comes from `processor_config.json` while the preprocessor values are kept.
    func testProcessorConfigFallsBackToProcessorConfigClass() async throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try MiniCPMVFixtures.mlxV47Preprocessor.write(
            to: dir.appending(component: "preprocessor_config.json"), atomically: true,
            encoding: .utf8)
        try MiniCPMVFixtures.mlxV47Processor.write(
            to: dir.appending(component: "processor_config.json"), atomically: true,
            encoding: .utf8)

        let (data, base) = try await loadProcessorConfig(from: dir)
        XCTAssertEqual(base.processorClass, "MiniCPMV4_7Processor")
        let config = try JSONDecoder.json5().decode(
            MiniCPMVProcessorConfiguration.self, from: data)
        XCTAssertEqual(config.maxSliceNums, 9)
        XCTAssertEqual(config.scaleResolution, 448)
        XCTAssertEqual(config.patchSize, 14)
        XCTAssertTrue(config.useImageId)
        XCTAssertTrue(config.sliceMode)
        XCTAssertEqual(config.imageMean, [0.5, 0.5, 0.5])
        XCTAssertEqual(config.imageStd, [0.5, 0.5, 0.5])
    }

    func testProcessorConfigV46Unchanged() async throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try MiniCPMVFixtures.openbmbV46Preprocessor.write(
            to: dir.appending(component: "preprocessor_config.json"), atomically: true,
            encoding: .utf8)
        let (_, base) = try await loadProcessorConfig(from: dir)
        XCTAssertEqual(base.processorClass, "MiniCPMV4_6Processor")
    }

    func testProcessorConfigWithoutAnyClassStillThrows() async throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try MiniCPMVFixtures.mlxV47Preprocessor.write(
            to: dir.appending(component: "preprocessor_config.json"), atomically: true,
            encoding: .utf8)
        do {
            _ = try await loadProcessorConfig(from: dir)
            XCTFail("expected a decoding error")
        } catch {}
    }

    // MARK: - Prompt layout

    func testV47PromptLayoutHasNoParenthesesAndValidCanvas() async throws {
        let tokenizer = CharTokenizer()
        let processor = try await makeProcessor("MiniCPMV4_7Processor", tokenizer: tokenizer)
        let input = try await processor.prepare(input: imageInput(prompt: "Hi"))
        let ids = input.text.tokens.reshaped(-1).asArray(Int32.self).map(Int.init)
        let frames = try XCTUnwrap(input.image?.frames)
        XCTAssertGreaterThan(frames.count, 1, "expected a sliced image")

        // <|im_start|>user\n<image_id>0</image_id><image>
        let head = [248_045] + tokenizer.encode(text: "user\n", addSpecialTokens: false)
        XCTAssertEqual(Array(ids.prefix(head.count)), head)
        XCTAssertEqual(ids[head.count], imageIdStart)
        XCTAssertFalse(ids.contains(tokenizer.id(of: "(")))
        XCTAssertFalse(ids.contains(tokenizer.id(of: ")")))

        // ...</slice>\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
        let tail =
            [sliceEnd] + tokenizer.encode(text: "\nHi", addSpecialTokens: false) + [248_046]
            + tokenizer.encode(text: "\n", addSpecialTokens: false) + [248_045]
            + tokenizer.encode(text: "assistant\n<think>\n\n</think>\n\n", addSpecialTokens: false)
        XCTAssertEqual(Array(ids.suffix(tail.count)), tail)

        // The processor output feeds the canvas computation without mismatches.
        let (positions, delta) = try MiniCPMVCanvasRoPE.positionIds(
            inputIds: ids, grids: frames.map { ($0.h, $0.w) }, markers: .minicpmV47,
            mergeFactor: 4)
        XCTAssertEqual(positions[0].count, ids.count)
        XCTAssertLessThan(delta, 0)
    }

    func testV46PromptLayoutUnchanged() async throws {
        let tokenizer = CharTokenizer()
        let processor = try await makeProcessor("MiniCPMV4_6Processor", tokenizer: tokenizer)
        let input = try await processor.prepare(input: imageInput(prompt: "Hi"))
        let ids = input.text.tokens.reshaped(-1).asArray(Int32.self).map(Int.init)

        let head = [248_045] + tokenizer.encode(text: "user\n(", addSpecialTokens: false)
        XCTAssertEqual(Array(ids.prefix(head.count)), head)
        XCTAssertEqual(ids[head.count], imageIdStart)
        let tail =
            [sliceEnd] + tokenizer.encode(text: ")\nHi", addSpecialTokens: false) + [248_046]
        XCTAssertTrue(
            ids.indices.contains { i in Array(ids[i...].prefix(tail.count)) == tail },
            "4.6 keeps the (<image>./</image>) wrapping")
    }

    /// 640x483 refines each of the 2x1 slices from 320x483 to 364x550, and 364 / 56 = 6.5.
    /// Python's `round` gives 6 (slice grid 40x24 patches, as mlx-vlm's 4.7 processor
    /// reports); the 4.6 layout keeps its original rounding (7 -> 40x28).
    func testSliceRoundingHalfToEvenFor47Only() async throws {
        let image = CIImage(color: CIColor(red: 0.2, green: 0.5, blue: 0.8))
            .cropped(to: CGRect(x: 0, y: 0, width: 640, height: 483))
        let input = UserInput(prompt: .text("Hi"), images: [.ciImage(image)])

        let v47 = try await makeProcessor("MiniCPMV4_7Processor", tokenizer: CharTokenizer())
        let prepared47 = try await v47.prepare(input: input)
        let frames47 = try XCTUnwrap(prepared47.image?.frames)
        XCTAssertEqual(frames47.map { [$0.h, $0.w] }, [[28, 36], [40, 24], [40, 24]])

        let v46 = try await makeProcessor("MiniCPMV4_6Processor", tokenizer: CharTokenizer())
        let prepared46 = try await v46.prepare(input: input)
        let frames46 = try XCTUnwrap(prepared46.image?.frames)
        XCTAssertEqual(frames46.map { [$0.h, $0.w] }, [[28, 36], [40, 28], [40, 28]])
    }

    // MARK: - Helpers

    private func csv(_ s: String) -> [Int] {
        s.split(separator: ",").map { Int($0)! }
    }

    private func makeTempDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appending(component: "minicpmv47-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func makeProcessor(_ type: String, tokenizer: any Tokenizer) async throws
        -> any UserInputProcessor
    {
        try await VLMProcessorTypeRegistry.shared.createModel(
            configuration: Data(MiniCPMVFixtures.mlxV47Preprocessor.utf8),
            processorType: type, tokenizer: tokenizer)
    }

    private func imageInput(prompt: String) -> UserInput {
        // 1000x700: large enough to be sliced (multiple > 1).
        let image = CIImage(color: CIColor(red: 0.2, green: 0.5, blue: 0.8))
            .cropped(to: CGRect(x: 0, y: 0, width: 1000, height: 700))
        return UserInput(prompt: .text(prompt), images: [.ciImage(image)])
    }
}

/// Character-level tokenizer: `\n` is 198 (as in the Qwen vocabulary), every other
/// character maps to `1000 + unicode scalar`.
private struct CharTokenizer: MLXLMCommon.Tokenizer {
    func id(of character: Character) -> Int {
        character == "\n" ? 198 : 1000 + Int(character.unicodeScalars.first!.value)
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        text.map(id(of:))
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}
