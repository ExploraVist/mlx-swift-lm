//
//  MiniCPMV46.swift
//  mlx-swift-lm
//
//  Port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/minicpmv4_6
//
//  MiniCPM-V 4.6 = SigLIP2-400M vision tower (+ VitMerger at layer 6 and a
//  final 2x2 Merger, 16x token downsample) + Qwen3.5-0.8B hybrid LLM
//  (reused from Qwen35.swift / Qwen35Language.Model).
//
//  Checkpoint: mlx-community/MiniCPM-V-4.6-4bit
//  Weight namespaces (already MLX-sanitized at conversion):
//    language_model.model.*   vision_tower.*   vit_merger.*   merger.mlp.*
//

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct MiniCPMV46Configuration: Codable, Sendable {

    public struct VisionConfiguration: Codable, Sendable {
        public var hiddenSize: Int = 1152
        public var intermediateSize: Int = 4304
        public var hiddenLayers: Int = 27
        public var attentionHeads: Int = 16
        public var channels: Int = 3
        public var imageSize: Int = 980
        public var patchSize: Int = 14
        public var layerNormEps: Float = 1e-6

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case channels = "num_channels"
            case imageSize = "image_size"
            case patchSize = "patch_size"
            case layerNormEps = "layer_norm_eps"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            self.hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1152
            self.intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4304
            self.hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 27
            self.attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
            self.channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? 3
            self.imageSize = try c.decodeIfPresent(Int.self, forKey: .imageSize) ?? 980
            self.patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
            self.layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
        }
    }

    public let textConfiguration: Qwen35Configuration.TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public var modelType: String = "minicpmv4_6"

    private let _imageTokenId: Int?
    public var imageTokenId: Int { _imageTokenId ?? 248_056 }
    private let _videoTokenId: Int?
    public var videoTokenId: Int { _videoTokenId ?? 248_057 }
    private let _insertLayerId: Int?
    public var insertLayerId: Int { _insertLayerId ?? 6 }
    private let _dropVisionLastLayer: Bool?
    public var dropVisionLastLayer: Bool { _dropVisionLastLayer ?? false }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case _imageTokenId = "image_token_id"
        case _videoTokenId = "video_token_id"
        case _insertLayerId = "insert_layer_id"
        case _dropVisionLastLayer = "drop_vision_last_layer"
    }
}

// MARK: - Neural Engine encoder hook

/// Supplies the vision encoder's layer stack from outside the model, so it can
/// run on the Apple Neural Engine via CoreML rather than on the GPU via MLX.
///
/// MLX cannot target the ANE — it is a Metal framework, and the ANE has no
/// public compute API at all, so CoreML is the only way in. That makes this a
/// hand-off rather than a backend: the app compiles the layers into CoreML
/// models and hands them back through this protocol.
///
/// Both calls take and return `[1, L, D]` in MLX's channels-last layout. The
/// conforming type is responsible for the transpose into the ANE's preferred
/// (B, C, 1, S) form and back.
public protocol LabsANEVisionEncoder: AnyObject {
    /// Encoder layers `0...insertLayerId`. Return nil to decline (falls back
    /// to MLX) — e.g. when `L` is not a length the model was compiled for.
    func runFront(_ x: MLXArray) -> MLXArray?
    /// Encoder layers `insertLayerId+1..<count`, followed by post_layernorm.
    func runBack(_ x: MLXArray) -> MLXArray?
}

// MARK: - Vision (SigLIP2)

private enum MiniCPMV46Vision {

    final class Attention: Module {
        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        let numHeads: Int
        let headDim: Int
        let scale: Float

        init(_ config: MiniCPMV46Configuration.VisionConfiguration) {
            let dim = config.hiddenSize
            self.numHeads = config.attentionHeads
            self.headDim = dim / numHeads
            self.scale = pow(Float(headDim), -0.5)
            _qProj.wrappedValue = Linear(dim, dim, bias: true)
            _kProj.wrappedValue = Linear(dim, dim, bias: true)
            _vProj.wrappedValue = Linear(dim, dim, bias: true)
            _outProj.wrappedValue = Linear(dim, dim, bias: true)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let (B, L, D) = (x.dim(0), x.dim(1), x.dim(2))
            let q = qProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
            let k = kProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
            let v = vProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
            let out = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: .none
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, D)
            return outProj(out)
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "fc1") var fc1: Linear
        @ModuleInfo(key: "fc2") var fc2: Linear
        let act = GELU(approximation: .precise)

        init(_ config: MiniCPMV46Configuration.VisionConfiguration) {
            _fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
            _fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: true)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(act(fc1(x)))
        }
    }

    final class EncoderLayer: Module {
        @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
        @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm
        @ModuleInfo(key: "self_attn") var selfAttn: Attention
        @ModuleInfo(key: "mlp") var mlp: MLP

        init(_ config: MiniCPMV46Configuration.VisionConfiguration) {
            _layerNorm1.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
            _layerNorm2.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
            _selfAttn.wrappedValue = Attention(config)
            _mlp.wrappedValue = MLP(config)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var h = x + selfAttn(layerNorm1(x))
            h = h + mlp(layerNorm2(h))
            return h
        }
    }

    final class Encoder: Module {
        @ModuleInfo(key: "layers") var layers: [EncoderLayer]

        init(_ config: MiniCPMV46Configuration.VisionConfiguration) {
            _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in EncoderLayer(config) }
            super.init()
        }
    }

    final class Embeddings: Module {
        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
        @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

        let patchSize: Int
        let embedDim: Int

        init(_ config: MiniCPMV46Configuration.VisionConfiguration) {
            self.patchSize = config.patchSize
            self.embedDim = config.hiddenSize
            _patchEmbedding.wrappedValue = Conv2d(
                inputChannels: config.channels,
                outputChannels: config.hiddenSize,
                kernelSize: IntOrPair(config.patchSize),
                stride: IntOrPair(config.patchSize),
                bias: true
            )
            let side = config.imageSize / config.patchSize
            _positionEmbedding.wrappedValue = Embedding(
                embeddingCount: side * side, dimensions: config.hiddenSize)
            super.init()
        }

        private var positionGridSide: Int {
            let count = positionEmbedding.weight.dim(0)
            let side = Int(Double(count).squareRoot().rounded())
            return side * side == count ? side : 0
        }

        /// Position ids for an (h, w) patch grid using MiniCPM's bucketing over
        /// the pretrained (side x side) grid. CPU computation, exact port of
        /// `_build_position_buckets`.
        private func bucketPositionIds(h: Int, w: Int) -> [Int32] {
            let side = positionGridSide
            guard side > 1 else { return Array(repeating: 0, count: h * w) }
            // boundaries = 1/side, 2/side, ..., (side-1)/side
            func buckets(_ n: Int) -> [Int32] {
                let n = max(n, 1)
                return (0 ..< n).map { i -> Int32 in
                    let frac = min(Float(i) / Float(n), 1.0 - 1e-6)
                    // count of boundaries b where frac >= b
                    return Int32(min(Int(frac * Float(side)), side - 1))
                }
            }
            let bh = buckets(h)
            let bw = buckets(w)
            var ids = [Int32]()
            ids.reserveCapacity(h * w)
            for y in 0 ..< h {
                for x in 0 ..< w {
                    ids.append(bh[y] * Int32(side) + bw[x])
                }
            }
            return ids
        }

        /// Packed patch strip -> embeddings. `strip` is [patchSize, n*patchSize, C]
        /// (HWC), `tgt` = (h, w) patch grid with h*w == n.
        func callAsFunction(strip: MLXArray, h: Int, w: Int) -> MLXArray {
            let p = patchSize
            let n = strip.dim(1) / p
            let c = strip.dim(2)
            // [p, n*p, C] -> [p, n, p, C] -> [n, p, p, C] -> [n, p*p*C]
            var patches = strip.reshaped(p, n, p, c).transposed(1, 0, 2, 3).reshaped(n, p * p * c)
            patches = patches.asType(patchEmbedding.weight.dtype)
            // conv weight [O, kh, kw, C] -> [O, kh*kw*C]
            let weight = patchEmbedding.weight.reshaped(embedDim, p * p * c)
            var embeddings = matmul(patches, weight.transposed(1, 0))
            if let bias = patchEmbedding.bias {
                embeddings = embeddings + bias
            }
            let posIds = MLXArray(bucketPositionIds(h: h, w: w))
            embeddings = embeddings + positionEmbedding(posIds)
            return embeddings  // [n, D]
        }
    }

    final class VisionModel: Module {
        @ModuleInfo(key: "embeddings") var embeddings: Embeddings
        @ModuleInfo(key: "encoder") var encoder: Encoder
        @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm

        init(_ config: MiniCPMV46Configuration.VisionConfiguration) {
            _embeddings.wrappedValue = Embeddings(config)
            _encoder.wrappedValue = Encoder(config)
            _postLayerNorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
            super.init()
        }
    }
}

// MARK: - Mergers

/// Simple multi-head cross attention with q/k/v/out projections (all biased).
private final class MiniCPMV46CrossAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    let numHeads: Int
    let headDim: Int
    let scale: Float

    init(dim: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.scale = pow(Float(headDim), -0.5)
        _qProj.wrappedValue = Linear(dim, dim, bias: true)
        _kProj.wrappedValue = Linear(dim, dim, bias: true)
        _vProj.wrappedValue = Linear(dim, dim, bias: true)
        _outProj.wrappedValue = Linear(dim, dim, bias: true)
        super.init()
    }

    func callAsFunction(_ queries: MLXArray, _ keys: MLXArray, _ values: MLXArray) -> MLXArray {
        let (B, Lq, D) = (queries.dim(0), queries.dim(1), queries.dim(2))
        let Lk = keys.dim(1)
        let q = qProj(queries).reshaped(B, Lq, numHeads, headDim).transposed(0, 2, 1, 3)
        let k = kProj(keys).reshaped(B, Lk, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(values).reshaped(B, Lk, numHeads, headDim).transposed(0, 2, 1, 3)
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .none
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, Lq, D)
        return outProj(out)
    }
}

/// Window merger inserted mid-encoder: 2x2 window self-attention + MLP merge.
/// Halves the grid in each dimension.
private final class MiniCPMV46VitMerger: Module {
    @ModuleInfo(key: "pre_norm") var preNorm: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttn: MiniCPMV46CrossAttention
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    let act = GELU(approximation: .precise)

    let groupH: Int
    let groupW: Int

    init(visionHiddenSize: Int, mergedHiddenSize: Int, numHeads: Int, group: (Int, Int) = (2, 2)) {
        self.groupH = group.0
        self.groupW = group.1
        let groupHidden = visionHiddenSize * groupH * groupW
        _preNorm.wrappedValue = LayerNorm(dimensions: groupHidden, eps: 1e-6)
        _selfAttn.wrappedValue = MiniCPMV46CrossAttention(
            dim: visionHiddenSize, numHeads: numHeads)
        _layerNorm1.wrappedValue = LayerNorm(dimensions: visionHiddenSize, eps: 1e-6)
        _linear1.wrappedValue = Linear(groupHidden, mergedHiddenSize, bias: true)
        _linear2.wrappedValue = Linear(mergedHiddenSize, visionHiddenSize, bias: true)
        super.init()
    }

    /// x: [gridH*gridW, D] -> ([gridH/2*gridW/2, D], gridH/2, gridW/2)
    func callAsFunction(_ x: MLXArray, gridH: Int, gridW: Int) -> (MLXArray, Int, Int) {
        let d = x.dim(-1)
        let mh = gridH / groupH
        let mw = gridW / groupW
        let groupTokens = groupH * groupW

        var windows = x.reshaped(gridH, gridW, d)
            .reshaped(mh, groupH, mw, groupW, d)
            .transposed(0, 2, 1, 3, 4)
            .reshaped(mh * mw, groupTokens, d)

        let normed = layerNorm1(windows)
        windows = windows + selfAttn(normed, normed, normed)

        let residual = windows.mean(axis: 1)  // [m, D]
        var merged = windows.reshaped(mh * mw, groupTokens * d)
        merged = preNorm(merged)
        merged = linear1(merged)
        merged = act(merged)
        merged = linear2(merged)
        return (merged + residual, mh, mw)
    }
}

private final class MiniCPMV46MergerBlock: Module, UnaryLayer {
    @ModuleInfo(key: "pre_norm") var preNorm: LayerNorm
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    let act = GELU(approximation: .precise)

    init(hiddenSize: Int, outSize: Int) {
        _preNorm.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: 1e-6)
        _linear1.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        _linear2.wrappedValue = Linear(hiddenSize, outSize, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(act(linear1(preNorm(x))))
    }
}

/// Final 2x2 spatial merge -> LLM hidden size.
private final class MiniCPMV46Merger: Module {
    @ModuleInfo(key: "mlp") var mlp: [MiniCPMV46MergerBlock]

    let kernelH: Int
    let kernelW: Int

    init(hiddenSize: Int, outSize: Int, mergerTimes: Int = 1, kernel: (Int, Int) = (2, 2)) {
        self.kernelH = kernel.0
        self.kernelW = kernel.1
        let mergeTokens = kernel.0 * kernel.1
        _mlp.wrappedValue = (0 ..< mergerTimes).map { i in
            MiniCPMV46MergerBlock(
                hiddenSize: hiddenSize * mergeTokens,
                outSize: i == mergerTimes - 1 ? outSize : hiddenSize
            )
        }
        super.init()
    }

    /// x: [gridH*gridW, D] -> ([gridH/2*gridW/2, outSize], h, w)
    func callAsFunction(_ x: MLXArray, gridH: Int, gridW: Int) -> (MLXArray, Int, Int) {
        var h = gridH
        var w = gridW
        var hidden = x
        for layer in mlp {
            let d = hidden.dim(-1)
            let mh = h / kernelH
            let mw = w / kernelW
            hidden = hidden.reshaped(h, w, d)
                .reshaped(mh, kernelH, mw, kernelW, d)
                .transposed(0, 2, 1, 3, 4)
                .reshaped(mh * mw, d * kernelH * kernelW)
            hidden = layer(hidden)
            h = mh
            w = mw
        }
        return (hidden, h, w)
    }
}

// MARK: - Language wrapper (reuses Qwen3.5 hybrid core)

private final class MiniCPMV46LanguageModel: Module {
    @ModuleInfo var model: Qwen35Language.Model
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let textConfig: Qwen35Configuration.TextConfiguration

    init(_ textConfig: Qwen35Configuration.TextConfiguration) {
        self.textConfig = textConfig
        self.model = Qwen35Language.Model(textConfig)
        if !textConfig.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                textConfig.hiddenSize, textConfig.vocabularySize, bias: false)
        }
        super.init()
    }

    var faIdx: Int { model.faIdx }

    /// MiniCPM-V injects vision via inputs_embeds; positions are plain
    /// sequential ids broadcast to the 3 MRoPE axes (see language.py).
    /// When true, only the LAST position is projected to vocabulary.
    ///
    /// Prefill exists to populate the cache; its logits are discarded. But
    /// the projection runs over every prompt position first, and with a
    /// 248,094-token vocabulary that tensor is enormous:
    ///
    ///     1322 tokens x 248,094 vocab x 2 B = 656 MB
    ///
    /// which matches the measured 714 MB gap between the llmLoad phase
    /// (746 MB) and the prefill peak (1460 MB). The KV cache is NOT the
    /// culprit — only 6 of 24 layers are full-attention, so it is ~16 MB.
    ///
    /// Slicing to the final position makes it 1 x 248,094 x 2 B = 0.5 MB.
    /// Decode is unaffected: it passes one token at a time, so last-position
    /// and all-positions are the same thing there.
    func callAsFunction(
        _ inputs: MLXArray,
        inputsEmbeds: MLXArray? = nil,
        cache: [KVCache?]? = nil,
        lastPositionOnly: Bool = false
    ) -> MLXArray {
        let inputs2d = inputs.ndim == 1 ? inputs.expandedDimensions(axis: 0) : inputs
        let batch = inputs2d.dim(0)
        let seqLen = inputs2d.dim(1)

        var offset = 0
        if let cache, let fa = cache[model.faIdx] {
            offset = fa.offset
        }

        var positions = MLXArray(offset ..< (offset + seqLen)).asType(.int32)
        positions = broadcast(positions[.newAxis, .newAxis, 0...], to: [3, batch, seqLen])

        var out = model(
            inputs2d, inputsEmbeds: inputsEmbeds, cache: cache, positionIds: positions)

        // Drop every position but the last BEFORE the vocabulary projection.
        if lastPositionOnly, out.dim(1) > 1 {
            out = out[0..., (out.dim(1) - 1)...]
        }

        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    func makeCache() -> [KVCache] {
        let interval = textConfig.fullAttentionInterval
        return (0 ..< textConfig.hiddenLayers).map { i in
            if (i % interval) == interval - 1 {
                return KVCacheSimple()
            }
            return MambaCache()
        }
    }
}

// MARK: - Main model

public class MiniCPMV46: Module, VLMModel {
    @ModuleInfo(key: "vision_tower") private var visionTower: MiniCPMV46Vision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: MiniCPMV46LanguageModel
    @ModuleInfo(key: "vit_merger") private var vitMerger: MiniCPMV46VitMerger
    @ModuleInfo(key: "merger") private var merger: MiniCPMV46Merger

    public let config: MiniCPMV46Configuration

    public init(_ config: MiniCPMV46Configuration) {
        self.config = config
        _visionTower.wrappedValue = MiniCPMV46Vision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = MiniCPMV46LanguageModel(config.textConfiguration)
        _vitMerger.wrappedValue = MiniCPMV46VitMerger(
            visionHiddenSize: config.visionConfiguration.hiddenSize,
            mergedHiddenSize: config.visionConfiguration.intermediateSize * 4,
            numHeads: config.visionConfiguration.attentionHeads
        )
        _merger.wrappedValue = MiniCPMV46Merger(
            hiddenSize: config.visionConfiguration.hiddenSize,
            outSize: config.textConfiguration.hiddenSize
        )
        super.init()
    }

    public var vocabularySize: Int { config.textConfiguration.vocabularySize }

    public var loraLayers: [Module] { [] }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.makeCache()
    }

    /// Per-encoder-layer hook for Labs weight streaming (Test 3).
    ///
    /// The vision tower is 823 MB of BF16 and, under dynamic residency, is
    /// the phase that sets the request peak. Streaming loads one layer at a
    /// time and frees it after use, so peak holds only the activations plus
    /// a couple of layers instead of the whole tower.
    ///
    /// `willUse` runs before layer `i` executes and must leave that layer's
    /// parameters attached; `didUse` runs after, and may release them. Both
    /// are nil in the normal path, so production behaviour is byte-identical
    /// — this compiles to two nil checks per layer.
    /// The layer module is handed over so the callee can attach/release
    /// parameters on THAT module alone. Updating the whole model from inside
    /// this loop mutates the tree being enumerated and walks every parameter
    /// in the model once per layer — unsafe, and slow enough to matter.
    public struct LabsLayerStream {
        public var willUse: (Int, Module) -> Void
        public var didUse: (Int, Module) -> Void
        /// Whether this layer's output must be materialized before didUse.
        /// Only true when weights are about to be released — MLX is lazy, so
        /// an eval is required then and only then.
        ///
        /// Previously eval() ran after EVERY layer regardless of chunk size:
        /// 27 GPU syncs where chunk 9 needs 3. That is almost certainly why
        /// chunk size never showed up in the timings — the sync cost was
        /// constant across settings and swamped the difference.
        public var needsEval: (Int) -> Bool
        /// Called once the embeddings have produced every strip's hidden
        /// state. `visionTower.embeddings` is not touched again this encode.
        public var didFinishEmbeddings: () -> Void = {}
        /// Called after the vitMerger pass at `insertLayerId` (which is 6 of
        /// 27), so its ~209 MB is dead for the remaining 21 layers rather
        /// than held until the encode ends.
        public var didFinishVitMerger: () -> Void = {}
        public init(willUse: @escaping (Int, Module) -> Void,
                    didUse: @escaping (Int, Module) -> Void,
                    needsEval: @escaping (Int) -> Bool = { _ in true },
                    didFinishEmbeddings: @escaping () -> Void = {},
                    didFinishVitMerger: @escaping () -> Void = {}) {
            self.willUse = willUse
            self.didUse = didUse
            self.needsEval = needsEval
            self.didFinishEmbeddings = didFinishEmbeddings
            self.didFinishVitMerger = didFinishVitMerger
        }
    }

    /// Batch same-shaped strips through each encoder layer instead of running
    /// them one at a time.
    ///
    /// processImageBlocks emits one overview strip plus grid.x*grid.y refined
    /// slices that are ALL exactly cellW x cellH, so a 9-strip image is 1
    /// overview + 8 identical slices. The 8 stack into [8, n, D] and go
    /// through a layer in a single call: 9 sequential matmuls become 2.
    /// Applies to the normal path as well as the streamed one.
    public var labsBatchStrips = false
    public var labsLayerStream: LabsLayerStream?

    /// An external implementation of the encoder layer stack, so Labs can run
    /// those layers on the Apple Neural Engine through CoreML instead of on
    /// the GPU through MLX.
    ///
    /// The split is at `insertLayerId` (6 of 27) for two reasons. iOS caps an
    /// ANE model at ~1 GB of weights — the same limit that makes Apple's own
    /// ml-stable-diffusion ship a `--chunk-unet` flag — and 27 layers is
    /// 823 MB, uncomfortably close. And the sequence length changes here
    /// anyway when vitMerger does its 2x2 merge (1024 -> 256), so cutting here
    /// makes each half statically shaped, which the ANE requires.
    ///
    /// vitMerger itself deliberately stays in MLX: its window regrouping is
    /// the op most likely to fail ANE compilation, and it is only 209 MB.
    ///
    /// Returning nil from either call falls the whole encode back to the MLX
    /// path — used when a strip's sequence length is not one the compiled
    /// model was built for.
    public var labsANEEncoder: (any LabsANEVisionEncoder)?

    /// Vision encode with the layer stack on the Neural Engine.
    ///
    /// MLX still does the embeddings, vitMerger and the final merger; the ANE
    /// does layers 0...insertLayerId and insertLayerId+1..<27 plus
    /// post_layernorm, which is 823 MB of the tower's 1097 MB and essentially
    /// all of its compute.
    private func visionFeaturesANE(
        strips: [(MLXArray, Int, Int)], encoder: any LabsANEVisionEncoder
    ) -> [MLXArray]? {
        var hidden: [MLXArray] = strips.map { (strip, h, w) in
            visionTower.embeddings(strip: strip, h: h, w: w)
                .expandedDimensions(axis: 0)
        }
        var gridH = strips.map(\.1)
        var gridW = strips.map(\.2)

        // Materialize every strip in ONE eval rather than one per strip.
        //
        // MLX is lazy: nothing runs until eval() or a read, and eval() blocks
        // the CPU until the GPU is done. Handing a strip to CoreML requires
        // reading it, so a per-strip eval means 9 blocking round-trips per pass
        // — 18 per image — with the GPU waking, doing one strip, then idling
        // while the ANE works. The strips are independent, so one submission
        // covers all of them and the GPU then stays out of the way while the
        // ANE runs uninterrupted.
        var staged = hidden.map { $0.asType(.float16) }
        eval(staged)
        // Embeddings are materialized, so the patch/position tables are dead
        // for the rest of the encode. The streamed path already signalled this;
        // the ANE path was holding all 274 MB of scaffolding to the end.
        labsLayerStream?.didFinishEmbeddings()

        for s in staged.indices {
            guard let out = encoder.runFront(staged[s]) else { return nil }
            hidden[s] = out
        }
        for s in hidden.indices {
            let (merged, mh, mw) = vitMerger(
                hidden[s][0], gridH: gridH[s], gridW: gridW[s])
            hidden[s] = merged.expandedDimensions(axis: 0)
            gridH[s] = mh
            gridW[s] = mw
        }
        staged = hidden.map { $0.asType(.float16) }
        eval(staged)
        // Every strip has been merged; vitMerger's 209 MB is dead before the
        // second ANE pass, which is the pass with the larger (610 MB) model
        // resident — so this is exactly where the headroom is worth most.
        labsLayerStream?.didFinishVitMerger()

        for s in staged.indices {
            guard let out = encoder.runBack(staged[s]) else { return nil }
            hidden[s] = out
        }
        // post_layernorm is already applied inside runBack.
        return hidden.indices.map { s in
            let (tokens, _, _) = merger(
                hidden[s][0], gridH: gridH[s], gridW: gridW[s])
            return tokens
        }
    }

    /// Number of vision encoder layers, so Labs can address them by index.
    public var labsVisionLayerCount: Int { visionTower.encoder.layers.count }

    /// Labs Test 3: the same computation as `visionFeatures`, with the loop
    /// nest INVERTED — layers outside, strips inside.
    ///
    /// visionFeatures runs per strip, so a 9-strip image walks all 27 encoder
    /// layers nine times. Under weight streaming that is 243 layer loads and
    /// ~7 GB of reads, and each load (~12 ms) has only one strip of compute
    /// (~6 ms) to hide behind — the prefetch can never get ahead. Measured:
    /// 1 hit / 242 misses, 3.1 s stalled, vision 1535 -> 4956 ms.
    ///
    /// Visiting each layer once and pushing every strip through it makes that
    /// 27 loads and ~823 MB, with ~57 ms of compute per 12 ms load. Identical
    /// arithmetic per strip — layers are stateless w.r.t. strips, so order of
    /// traversal cannot change the result, which the parity check confirms.
    ///
    /// vitMerger fires mid-stack at `insertLayerId` and rewrites each strip's
    /// grid, so grids are carried per strip rather than as a single value.
    private func visionFeaturesStreamed(strips: [(MLXArray, Int, Int)]) -> [MLXArray] {
        var hidden: [MLXArray] = strips.map { (strip, h, w) in
            visionTower.embeddings(strip: strip, h: h, w: w)
                .expandedDimensions(axis: 0)
        }
        var gridH = strips.map(\.1)
        var gridW = strips.map(\.2)

        if let stream = labsLayerStream {
            eval(hidden)                    // embeddings output materialized
            stream.didFinishEmbeddings()    // ~13 MB, dead from here
        }

        // Strips that share a token count can go through a layer together.
        // Grouped up front so the grouping cost is paid once, not per layer.
        let groups: [[Int]] = labsBatchStrips
            ? Dictionary(grouping: hidden.indices) { hidden[$0].dim(1) }
                .values.map { $0.sorted() }.sorted { $0[0] < $1[0] }
            : hidden.indices.map { [$0] }

        for (index, layer) in visionTower.encoder.layers.enumerated() {
            labsLayerStream?.willUse(index, layer)

            for group in groups {
                if group.count == 1 {
                    let s = group[0]
                    hidden[s] = layer(hidden[s])
                } else {
                    // [g, n, D] in one call instead of g calls.
                    let batch = concatenated(group.map { hidden[$0] }, axis: 0)
                    let out = layer(batch)
                    for (k, s) in group.enumerated() {
                        hidden[s] = out[k ..< (k + 1)]
                    }
                }
                if index == config.insertLayerId {
                    // vitMerger is per-strip (it rewrites each grid), so the
                    // batch is split here and regrouped on the next layer.
                    for s in group {
                        let (merged, mh, mw) = vitMerger(
                            hidden[s][0], gridH: gridH[s], gridW: gridW[s])
                        hidden[s] = merged.expandedDimensions(axis: 0)
                        gridH[s] = mh
                        gridW[s] = mw
                    }
                }
            }

            if index == config.insertLayerId, let stream = labsLayerStream {
                // Every strip has been merged; vitMerger has no further use
                // this encode. At insertLayerId = 6 of 27 that frees ~209 MB
                // for the remaining 21 layers — exactly the window where the
                // LLM prefetch lands.
                eval(hidden)
                stream.didFinishVitMerger()
            }

            if let stream = labsLayerStream {
                // Materialize only when this layer is about to be released.
                if stream.needsEval(index) { eval(hidden) }
                stream.didUse(index, layer)
            }
        }

        return hidden.indices.map { s in
            let h = visionTower.postLayerNorm(hidden[s])[0]
            let (tokens, _, _) = merger(h, gridH: gridH[s], gridW: gridW[s])
            return tokens
        }
    }

    /// Run one packed slice through the vision tower (VitMerger inserted at
    /// `insertLayerId`) and the final Merger. Returns [tokens, llmHidden].
    private func visionFeatures(strip: MLXArray, h: Int, w: Int) -> MLXArray {
        var hidden = visionTower.embeddings(strip: strip, h: h, w: w)
        hidden = hidden.expandedDimensions(axis: 0)  // [1, n, D]

        var gridH = h
        var gridW = w
        for (index, layer) in visionTower.encoder.layers.enumerated() {
            hidden = layer(hidden)
            if index == config.insertLayerId {
                let (merged, mh, mw) = vitMerger(hidden[0], gridH: gridH, gridW: gridW)
                hidden = merged.expandedDimensions(axis: 0)
                gridH = mh
                gridW = mw
            }
        }
        hidden = visionTower.postLayerNorm(hidden)[0]

        let (tokens, _, _) = merger(hidden, gridH: gridH, gridW: gridW)
        return tokens
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], windowSize _: Int?
    ) throws -> PrepareResult {
        var inputIds = input.text.tokens
        if inputIds.ndim == 1 {
            inputIds = inputIds.expandedDimensions(axis: 0)
        }

        var inputEmbeddings: MLXArray? = nil

        if let image = input.image, let frames = image.frames, !frames.isEmpty {
            let textEmbeds = languageModel.model.embedTokens(inputIds)

            // pixels: [1, patch, totalWidth, C] — slices concatenated on width.
            var pixels = image.pixels
            if pixels.ndim == 4 {
                pixels = pixels[0]
            }
            let patch = pixels.dim(0)

            var features: [MLXArray] = []
            var cursor = 0
            for frame in frames {
                let h = frame.h
                let w = frame.w
                let widthPx = h * w * patch
                let strip = pixels[0..., cursor ..< (cursor + widthPx), 0...]
                cursor += widthPx
                features.append(visionFeatures(strip: strip, h: h, w: w))
            }
            let imageFeatures = concatenated(features, axis: 0)
                .asType(textEmbeds.dtype)

            inputEmbeddings = try scatterImageFeatures(
                features: imageFeatures,
                embeds: textEmbeds,
                inputIds: inputIds,
                placeholderId: config.imageTokenId
            )
        }

        let typedCache: [KVCache?]? = cache.isEmpty ? nil : cache.map { $0 }
        let logits = languageModel(
            inputIds, inputsEmbeds: inputEmbeddings, cache: typedCache)
        return .logits(LMOutput(logits: logits))
    }

    public func callAsFunction(
        _ input: LMInput.Text, cache: [any KVCache]?, state _: LMOutput.State?
    ) -> LMOutput {
        let typedCache: [KVCache?]? = cache.map { c in c.map { $0 } }
        let logits = languageModel(input.tokens, inputsEmbeds: nil, cache: typedCache)
        return LMOutput(logits: logits)
    }

    private func scatterImageFeatures(
        features: MLXArray, embeds: MLXArray, inputIds: MLXArray, placeholderId: Int
    ) throws -> MLXArray {
        let ids = inputIds[0].asArray(Int32.self)
        var indices: [UInt32] = []
        indices.reserveCapacity(features.dim(0))
        for (i, id) in ids.enumerated() where id == Int32(placeholderId) {
            indices.append(UInt32(i))
        }
        guard indices.count == features.dim(0) else {
            throw MiniCPMV46Error.placeholderMismatch(
                placeholders: indices.count, features: features.dim(0))
        }
        var flat = embeds[0]
        flat[MLXArray(indices)] = features
        return flat.expandedDimensions(axis: 0)
    }

    // MARK: - Labs split execution (ExploraVist execution-schedule experiments)
    //
    // `prepare(_:cache:windowSize:)` fuses vision encode and the LLM forward.
    // These two calls expose the same computation split at the natural
    // boundary so a caller can (a) time the halves separately and (b) release
    // the vision half after `labsEncodeImage` returns — nothing below the
    // split touches vision weights, and nothing above touches LLM weights.

    /// Vision half only: tower + VitMerger + Merger over the input's packed
    /// strips. Returns concatenated features [nImageTokens, llmHidden],
    /// evaluated (materialized) so the vision weights may be released
    /// immediately afterwards.
    public func labsEncodeImage(_ input: LMInput) throws -> MLXArray {
        guard let image = input.image, let frames = image.frames, !frames.isEmpty else {
            throw MiniCPMV46Error.imageProcessingFailed("labsEncodeImage: input has no image")
        }
        var pixels = image.pixels
        if pixels.ndim == 4 {
            pixels = pixels[0]
        }
        let patch = pixels.dim(0)

        // Slice first, then dispatch. With streaming on we take the inverted
        // path so each encoder layer is loaded once for ALL strips instead of
        // once per strip — 27 loads rather than 243.
        var strips: [(MLXArray, Int, Int)] = []
        var cursor = 0
        for frame in frames {
            let h = frame.h
            let w = frame.w
            let widthPx = h * w * patch
            strips.append((pixels[0..., cursor ..< (cursor + widthPx), 0...], h, w))
            cursor += widthPx
        }

        // Streaming implies the inverted loop. Batching alone also uses it,
        // since batching same-shaped strips is only expressible with strips
        // on the inside — visionFeatures handles exactly one strip.
        // The ANE path is tried first when one is attached, and falls through
        // to MLX if it declines (unsupported sequence length).
        let features: [MLXArray] =
            labsANEEncoder.flatMap { visionFeaturesANE(strips: strips, encoder: $0) }
            ?? ((labsLayerStream != nil || labsBatchStrips)
                ? visionFeaturesStreamed(strips: strips)
                : strips.map { visionFeatures(strip: $0.0, h: $0.1, w: $0.2) })

        let imageFeatures = concatenated(features, axis: 0)
        eval(imageFeatures)
        return imageFeatures
    }

    /// Vision and prefill, OVERLAPPED.
    ///
    /// The ANE and the GPU are separate silicon, and today exactly one of them
    /// is working at any moment: the GPU idles through the entire ANE encode,
    /// then the ANE idles through prefill. Serially that is vision + prefill;
    /// overlapped it is max(vision, prefill).
    ///
    /// The overlap needs no threads and no extra queues, because **MLX is
    /// lazy**. Issuing prefill operations without eval() queues them on the GPU
    /// and returns immediately; the next CoreML prediction then blocks on the
    /// ANE while the GPU works through that queue. Only the final eval waits.
    ///
    /// Correctness is the same argument as chunked prefill: image tokens are a
    /// contiguous, ordered block in the prompt, attention is causal, and the
    /// cache is built left to right. Once strip s is encoded, every prompt
    /// position up to the end of its `<|image_pad|>` run is known, and
    /// prefilling it is exactly as valid as prefilling the whole prompt at
    /// once.
    ///
    /// Requires an attached `labsANEEncoder`; without one there is nothing to
    /// overlap with and this throws so the caller uses the normal path.
    ///
    /// - Parameter stripsPerChunk: strips encoded before a prefill is issued.
    ///   Smaller means finer overlap and worse GPU efficiency (one matmul over
    ///   N positions beats several over N/k); larger means the reverse.
    public func labsEncodeAndPrefillOverlapped(
        _ input: LMInput, cache: [any KVCache], stripsPerChunk: Int = 3
    ) throws -> MLXArray {
        guard let encoder = labsANEEncoder else {
            throw MiniCPMV46Error.imageProcessingFailed(
                "labsEncodeAndPrefillOverlapped: no ANE encoder attached")
        }
        guard let image = input.image, let frames = image.frames, !frames.isEmpty else {
            throw MiniCPMV46Error.imageProcessingFailed("no image")
        }

        var pixels = image.pixels
        if pixels.ndim == 4 { pixels = pixels[0] }
        let patch = pixels.dim(0)
        var strips: [(MLXArray, Int, Int)] = []
        var cursor = 0
        for frame in frames {
            let widthPx = frame.h * frame.w * patch
            strips.append((pixels[0..., cursor ..< (cursor + widthPx), 0...], frame.h, frame.w))
            cursor += widthPx
        }

        var inputIds = input.text.tokens
        if inputIds.ndim == 1 { inputIds = inputIds.expandedDimensions(axis: 0) }
        let total = inputIds.dim(1)
        let textEmbeds = languageModel.model.embedTokens(inputIds)
        let typedCache: [KVCache?]? = cache.isEmpty ? nil : cache.map { $0 }

        // Prompt positions holding an image placeholder, in order. The s-th
        // strip's tokens land in a contiguous run of these.
        let idsHost = inputIds[0].asArray(Int32.self)
        var padAt: [Int] = []
        padAt.reserveCapacity(idsHost.count)
        for (i, t) in idsHost.enumerated() where t == Int32(config.imageTokenId) {
            padAt.append(i)
        }

        // ---- vision: embeddings, ANE front, vitMerger (all strips) ----
        var hidden: [MLXArray] = strips.map { (strip, h, w) in
            visionTower.embeddings(strip: strip, h: h, w: w).expandedDimensions(axis: 0)
        }
        var gridH = strips.map(\.1)
        var gridW = strips.map(\.2)

        var staged = hidden.map { $0.asType(.float16) }
        eval(staged)
        labsLayerStream?.didFinishEmbeddings()
        for s in staged.indices {
            guard let out = encoder.runFront(staged[s]) else {
                throw MiniCPMV46Error.imageProcessingFailed(
                    "ANE declined a strip; use the non-overlapped path")
            }
            hidden[s] = out
        }
        for s in hidden.indices {
            let (merged, mh, mw) = vitMerger(hidden[s][0], gridH: gridH[s], gridW: gridW[s])
            hidden[s] = merged.expandedDimensions(axis: 0)
            gridH[s] = mh
            gridW[s] = mw
        }
        staged = hidden.map { $0.asType(.float16) }
        eval(staged)
        labsLayerStream?.didFinishVitMerger()

        // ---- ANE back pass, interleaved with prefill on the GPU ----
        var features: [MLXArray] = []
        var featuresFlushed = 0   // features already scattered into the cache
        var padsDone = 0          // image tokens produced so far
        var padsPrefilled = 0     // image tokens already in the cache
        var promptDone = 0        // prompt positions already in the cache

        func flush(upToPad pad: Int) throws {
            guard pad > padsPrefilled, pad <= padAt.count,
                  featuresFlushed < features.count else { return }
            let end = padAt[pad - 1] + 1
            guard end > promptDone else { return }
            let ids = inputIds[0..., promptDone ..< end]
            let embeds = textEmbeds[0..., promptDone ..< end]
            // Everything appended since the last flush covers exactly the pads
            // between padsPrefilled and pad — features are produced in strip
            // order and each strip fills a contiguous run.
            let feats = concatenated(
                Array(features[featuresFlushed ..< features.count]), axis: 0)
            let scattered = try scatterImageFeatures(
                features: feats.asType(textEmbeds.dtype), embeds: embeds,
                inputIds: ids, placeholderId: config.imageTokenId)
            // NO eval — this is the whole point. The work queues on the GPU and
            // the next ANE prediction runs against it.
            _ = languageModel(ids, inputsEmbeds: scattered, cache: typedCache,
                              lastPositionOnly: true)
            featuresFlushed = features.count
            padsPrefilled = pad
            promptDone = end
        }

        for s in staged.indices {
            guard let out = encoder.runBack(staged[s]) else {
                throw MiniCPMV46Error.imageProcessingFailed(
                    "ANE declined a strip; use the non-overlapped path")
            }
            let h = visionTower.postLayerNorm(out)[0]
            let (tokens, _, _) = merger(h, gridH: gridH[s], gridW: gridW[s])
            features.append(tokens)
            padsDone += tokens.dim(0)

            if (s + 1) % max(stripsPerChunk, 1) == 0 {
                try flush(upToPad: padsDone)
            }
        }
        try flush(upToPad: padsDone)

        // Tail: everything after the last image token (question, assistant
        // preamble). No image features involved.
        if promptDone < total {
            let ids = inputIds[0..., promptDone ..< total]
            let embeds = textEmbeds[0..., promptDone ..< total]
            _ = languageModel(ids, inputsEmbeds: embeds, cache: typedCache,
                              lastPositionOnly: true)
        }
        eval(cache)

        let imageFeatures = concatenated(features, axis: 0)
        eval(imageFeatures)
        return imageFeatures
    }

    /// LLM half only: embed tokens, scatter `imageFeatures` into the
    /// `<|image_pad|>` positions (when given), and run one forward pass into
    /// `cache`, discarding logits (prefill-only). The cache is evaluated
    /// before returning. Decode by passing the follow-on tokens plus this
    /// cache to `generateTokens`/`TokenIterator`.
    /// - Parameter chunkSize: prompt positions per forward pass. 0 (default)
    ///   is one pass over the whole prompt, exactly as before.
    ///
    ///   Chunking is mathematically identical: attention is causal and the
    ///   cache is built left to right, so tokens 0..<k can be prefilled before
    ///   k exists. Every chunk still attends to everything before it, already
    ///   in the cache.
    ///
    ///   On its own it is slightly SLOWER — one matmul over 3114 positions
    ///   beats six over 512 — so this is not a latency win by itself. It exists
    ///   so prefill can be overlapped with vision: image tokens are a
    ///   contiguous block, so a strip's tokens can be prefilled on the GPU
    ///   while the ANE encodes the next strip. At 36 slices that is
    ///   max(1188, 1564) instead of 1188 + 1564.
    ///
    ///   Chunk coarsely. Per-strip (64 positions) gives the most overlap and
    ///   the worst GPU efficiency; 4-8 strips keeps nearly all the overlap.
    public func labsPrefill(
        _ input: LMInput, imageFeatures: MLXArray?, cache: [any KVCache],
        chunkSize: Int = 0
    ) throws {
        var inputIds = input.text.tokens
        if inputIds.ndim == 1 {
            inputIds = inputIds.expandedDimensions(axis: 0)
        }

        var inputEmbeddings: MLXArray? = nil
        if let imageFeatures {
            let textEmbeds = languageModel.model.embedTokens(inputIds)
            inputEmbeddings = try scatterImageFeatures(
                features: imageFeatures.asType(textEmbeds.dtype),
                embeds: textEmbeds,
                inputIds: inputIds,
                placeholderId: config.imageTokenId
            )
        }

        let typedCache: [KVCache?]? = cache.isEmpty ? nil : cache.map { $0 }
        let total = inputIds.dim(1)
        let step = chunkSize > 0 ? min(chunkSize, total) : total

        var start = 0
        while start < total {
            let end = min(start + step, total)
            // The cache carries its own offset, so each pass appends at the
            // right positions — the same mechanism decode uses.
            let ids = inputIds[0..., start ..< end]
            let embeds = inputEmbeddings.map { $0[0..., start ..< end] }
            // Prefill discards the logits, so do not materialize 656 MB of them.
            _ = languageModel(
                ids, inputsEmbeds: embeds, cache: typedCache,
                lastPositionOnly: true)
            eval(cache)
            start = end
        }
    }

    public func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String:
        MLXArray]
    {
        if metadata["format"]?.lowercased() == "mlx" {
            return weights
        }
        return sanitize(weights: weights)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // mlx-community checkpoints are already in the target namespace; this
        // path only handles raw HF checkpoints (vpm./llm. style).
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)

        let normSuffixes = [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        ]

        for (originalKey, originalValue) in weights {
            var key = originalKey
            var value = originalValue

            if key.hasPrefix("model.") {
                key = String(key.dropFirst("model.".count))
            }
            if key.hasPrefix("llm.") {
                key = "language_model.model." + key.dropFirst("llm.".count)
            } else if key.hasPrefix("vpm.") {
                key = "vision_tower." + key.dropFirst("vpm.".count)
            } else if key.hasPrefix("language_model."), !key.hasPrefix("language_model.model.") {
                key = "language_model.model." + key.dropFirst("language_model.".count)
            } else if key.hasPrefix("lm_head.") {
                key = "language_model." + key
            }

            if key.contains("position_ids") { continue }

            if key.contains("conv1d.weight"), value.ndim == 3, value.dim(-1) != 1 {
                value = value.movedAxis(source: 2, destination: 1)
            }
            if key.hasSuffix("embeddings.patch_embedding.weight"), value.ndim == 4,
                value.dim(-1) != 3
            {
                value = value.transposed(0, 2, 3, 1)
            }
            if normSuffixes.contains(where: { key.hasSuffix($0) }), value.ndim == 1 {
                value = value + MLXArray(1, dtype: value.dtype)
            }

            sanitized[key] = value
        }

        if config.textConfiguration.tieWordEmbeddings {
            sanitized["language_model.lm_head.weight"] = nil
        }
        return sanitized
    }
}

enum MiniCPMV46Error: LocalizedError {
    case placeholderMismatch(placeholders: Int, features: Int)
    case imageProcessingFailed(String)

    var errorDescription: String? {
        switch self {
        case .placeholderMismatch(let p, let f):
            return "MiniCPM-V image placeholder span (\(p)) != vision feature count (\(f))"
        case .imageProcessingFailed(let msg):
            return "MiniCPM-V image processing failed: \(msg)"
        }
    }
}

// MARK: - Processor

public struct MiniCPMVProcessorConfiguration: Codable, Sendable {
    public var maxSliceNums: Int = 9
    public var scaleResolution: Int = 448
    public var patchSize: Int = 14
    public var useImageId: Bool = true
    public var sliceMode: Bool = true
    public var imageMean: [CGFloat] = [0.5, 0.5, 0.5]
    public var imageStd: [CGFloat] = [0.5, 0.5, 0.5]

    enum CodingKeys: String, CodingKey {
        case maxSliceNums = "max_slice_nums"
        case scaleResolution = "scale_resolution"
        case patchSize = "patch_size"
        case useImageId = "use_image_id"
        case sliceMode = "slice_mode"
        case imageMean = "image_mean"
        case imageStd = "image_std"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.maxSliceNums = try c.decodeIfPresent(Int.self, forKey: .maxSliceNums) ?? 9
        self.scaleResolution = try c.decodeIfPresent(Int.self, forKey: .scaleResolution) ?? 448
        self.patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
        self.useImageId = try c.decodeIfPresent(Bool.self, forKey: .useImageId) ?? true
        self.sliceMode = try c.decodeIfPresent(Bool.self, forKey: .sliceMode) ?? true
        self.imageMean = try c.decodeIfPresent([CGFloat].self, forKey: .imageMean) ?? [
            0.5, 0.5, 0.5,
        ]
        self.imageStd = try c.decodeIfPresent([CGFloat].self, forKey: .imageStd) ?? [0.5, 0.5, 0.5]
    }
}

public final class MiniCPMVProcessor: UserInputProcessor {

    private let config: MiniCPMVProcessorConfiguration
    private let tokenizer: any Tokenizer

    // Special token ids (MiniCPM-V 4.6 vocabulary)
    private let imStartId = 248_045  // <|im_start|>
    private let imEndId = 248_046  // <|im_end|>
    private let imageStartId = 248_078  // <image>
    private let imageEndId = 248_079  // </image>
    private let sliceStartId = 248_088  // <slice>
    private let sliceEndId = 248_089  // </slice>
    private let imageIdStartId = 248_090  // <image_id>
    private let imageIdEndId = 248_091  // </image_id>
    private let imagePadId = 248_056  // <|image_pad|> (fill token)
    private let tokenDivisor = 16  // downsample_mode "16x"

    public init(_ config: MiniCPMVProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    // MARK: image slicing math (exact port of processing_minicpmv4_6.py)

    private func ensureDivide(_ length: Int, _ patchSize: Int) -> Int {
        max(Int((Double(length) / Double(patchSize)).rounded()) * patchSize, patchSize)
    }

    private func findBestResize(
        _ size: (w: Int, h: Int), scaleResolution: Int, patchSize: Int, allowUpscale: Bool
    ) -> (w: Int, h: Int) {
        var width = size.w
        var height = size.h
        if width * height > scaleResolution * scaleResolution || allowUpscale {
            let ratio = Double(width) / Double(max(height, 1))
            height = Int(Double(scaleResolution) / max(ratio, 1e-6).squareRoot())
            width = Int(Double(height) * ratio)
        }
        let mergeFactor = patchSize * 4
        return (ensureDivide(width, mergeFactor), ensureDivide(height, mergeFactor))
    }

    private func getRefineSize(
        _ size: (w: Int, h: Int), grid: (x: Int, y: Int), scaleResolution: Int, patchSize: Int,
        allowUpscale: Bool
    ) -> (w: Int, h: Int) {
        let refineWidth = ensureDivide(size.w, grid.x)
        let refineHeight = ensureDivide(size.h, grid.y)
        let gridW = Double(refineWidth) / Double(grid.x)
        let gridH = Double(refineHeight) / Double(grid.y)
        let best = findBestResize(
            (Int(gridW), Int(gridH)), scaleResolution: scaleResolution, patchSize: patchSize,
            allowUpscale: allowUpscale)
        return (best.w * grid.x, best.h * grid.y)
    }

    private func getSlicedGrid(_ size: (w: Int, h: Int), maxSliceNums: Int) -> (x: Int, y: Int)? {
        let ratio =
            Double(size.w * size.h) / Double(config.scaleResolution * config.scaleResolution)
        let multiple = min(Int(ratio.rounded(.up)), maxSliceNums)
        if multiple <= 1 { return nil }

        var candidates: [Int] = []
        for gridNum in [multiple - 1, multiple, multiple + 1] {
            if gridNum == 1 || gridNum > maxSliceNums { continue }
            candidates.append(gridNum)
        }

        var grids: [(Int, Int)] = []
        for gridNum in candidates {
            var factor = 1
            while factor <= gridNum {
                if gridNum % factor == 0 {
                    grids.append((factor, gridNum / factor))
                }
                factor += 1
            }
        }

        let logRatio = log(Double(size.w) / Double(max(size.h, 1)))
        var best = (1, 1)
        var minError = Double.infinity
        for grid in grids {
            let error = abs(logRatio - log(Double(grid.0) / Double(grid.1)))
            if error < minError {
                best = grid
                minError = error
            }
        }
        return best
    }

    // MARK: image tensor helpers

    /// Bicubic resize + normalize + pack into a [patch, n*patch, C] strip.
    /// Returns (strip, h, w) with h,w in patch units.
    private func packedStrip(_ image: CIImage, size: (w: Int, h: Int)) throws -> (
        MLXArray, Int, Int
    ) {
        let resized = MediaProcessing.resampleBicubic(
            image, to: CGSize(width: size.w, height: size.h))
        // [1, C, H, W] float32 0...1
        var array = MediaProcessing.asMLXArray(resized)
        array = (array - 0.5) / 0.5
        // -> [H, W, C]
        array = array[0].transposed(1, 2, 0)

        let p = config.patchSize
        let h = size.h / p
        let w = size.w / p
        // crop any residual (sizes are multiples of 56 so this is a no-op)
        array = array[0 ..< (h * p), 0 ..< (w * p), 0...]
        // [H, W, C] -> [h, p, w, p, C] -> [p, h, w, p, C] -> [p, h*w*p, C]
        let c = array.dim(2)
        let strip = array.reshaped(h, p, w, p, c)
            .transposed(1, 0, 2, 3, 4)
            .reshaped(p, h * w * p, c)
        return (strip, h, w)
    }

    private func encode(_ text: String) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: false)
    }

    // MARK: prompt extraction

    private func extractText(_ input: UserInput) -> String {
        switch input.prompt {
        case .text(let text):
            return text
        case .messages(let messages):
            let userTexts = messages.compactMap { message -> String? in
                guard message["role"] as? String == "user" else { return nil }
                return message["content"] as? String
            }
            return userTexts.last ?? ""
        case .chat(let messages):
            let userTexts = messages.filter { $0.role == .user }.map { $0.content }
            return userTexts.last ?? ""
        }
    }

    // MARK: UserInputProcessor

    public func prepare(input: UserInput) async throws -> LMInput {
        let prompt = extractText(input)

        // Text-only path
        if input.images.isEmpty {
            var ids: [Int] = [imStartId]
            ids += encode("user\n" + prompt)
            ids += [imEndId]
            ids += encode("\n")
            ids += [imStartId]
            ids += encode("assistant\n<think>\n\n</think>\n\n")
            return LMInput(tokens: MLXArray(ids.map { Int32($0) }).expandedDimensions(axis: 0))
        }

        // Image path (single image supported; first image used)
        let (strips, frames, grid) = try processImageBlocks(input.images[0], processing: input.processing)

        // Assemble token ids:
        // <|im_start|>user\n ( [image blocks] )\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
        var ids: [Int] = [imStartId]
        ids += encode("user\n(")
        ids += imageBlockIds(frames: frames, grid: grid)
        ids += encode(")\n" + prompt)
        ids += [imEndId]
        ids += encode("\n")
        ids += [imStartId]
        ids += encode("assistant\n<think>\n\n</think>\n\n")

        let pixels = concatenated(strips, axis: 1).expandedDimensions(axis: 0)

        return LMInput(
            text: .init(tokens: MLXArray(ids.map { Int32($0) }).expandedDimensions(axis: 0)),
            image: .init(pixels: pixels, frames: frames)
        )
    }

    /// Slice the image and pack per-frame strips. Behavior identical to the
    /// former inline body of `prepare(input:)`.
    private func processImageBlocks(
        _ image: UserInput.Image, processing: UserInput.Processing?
    ) throws -> (strips: [MLXArray], frames: [THW], grid: (x: Int, y: Int)?) {
        let ciImage = try image.asCIImage()
        let oriented = MediaProcessing.apply(ciImage, processing: processing)
        let size = (
            w: Int(oriented.extent.width.rounded()), h: Int(oriented.extent.height.rounded())
        )

        var strips: [MLXArray] = []
        var frames: [THW] = []

        let maxSliceNums = config.maxSliceNums
        let grid = config.sliceMode ? getSlicedGrid(size, maxSliceNums: maxSliceNums) : nil

        if let grid {
            // Overview image (no upscale) + refined slices.
            let bestResize = findBestResize(
                size, scaleResolution: config.scaleResolution, patchSize: config.patchSize,
                allowUpscale: false)
            let (overview, oh, ow) = try packedStrip(oriented, size: bestResize)
            strips.append(overview)
            frames.append(THW(1, oh, ow))

            let refineSize = getRefineSize(
                size, grid: grid, scaleResolution: config.scaleResolution,
                patchSize: config.patchSize, allowUpscale: true)
            let refined = MediaProcessing.resampleBicubic(
                oriented, to: CGSize(width: refineSize.w, height: refineSize.h))
            let cellW = refineSize.w / grid.x
            let cellH = refineSize.h / grid.y
            for row in 0 ..< grid.y {
                for col in 0 ..< grid.x {
                    // CIImage origin is bottom-left; PIL crop boxes are top-left.
                    let top = row * cellH
                    let rect = CGRect(
                        x: CGFloat(col * cellW),
                        y: CGFloat(refineSize.h - top - cellH),
                        width: CGFloat(cellW),
                        height: CGFloat(cellH)
                    )
                    let cell = refined.cropped(to: rect)
                        .transformed(by: CGAffineTransform(
                            translationX: -rect.origin.x, y: -rect.origin.y))
                    let (strip, sh, sw) = try packedStrip(cell, size: (cellW, cellH))
                    strips.append(strip)
                    frames.append(THW(1, sh, sw))
                }
            }
        } else {
            let bestResize = findBestResize(
                size, scaleResolution: config.scaleResolution, patchSize: config.patchSize,
                allowUpscale: true)
            let (overview, oh, ow) = try packedStrip(oriented, size: bestResize)
            strips.append(overview)
            frames.append(THW(1, oh, ow))
        }

        return (strips, frames, grid)
    }

    /// The `<image_id>…</image_id><image>…</image><slice>…</slice>` id block
    /// between "user\n(" and ")\n". Behavior identical to the former inline
    /// body of `prepare(input:)`.
    private func imageBlockIds(frames: [THW], grid: (x: Int, y: Int)?) -> [Int] {
        var ids: [Int] = []

        // <image_id>0</image_id>
        if config.useImageId {
            ids += [imageIdStartId]
            ids += encode("0")
            ids += [imageIdEndId]
        }

        // overview: <image> pads </image>
        let overviewTokens = (frames[0].h * frames[0].w) / tokenDivisor
        ids += [imageStartId]
        ids += Array(repeating: imagePadId, count: overviewTokens)
        ids += [imageEndId]

        // slices: rows of <slice> pads </slice>, newline between rows
        if let grid, frames.count > 1 {
            let sliceTokens = (frames[1].h * frames[1].w) / tokenDivisor
            let newlineIds = encode("\n")
            for row in 0 ..< grid.y {
                for _ in 0 ..< grid.x {
                    ids += [sliceStartId]
                    ids += Array(repeating: imagePadId, count: sliceTokens)
                    ids += [sliceEndId]
                }
                if row != grid.y - 1 {
                    ids += newlineIds
                }
            }
        }

        return ids
    }

    // MARK: - Labs split-prompt API (ExploraVist execution-schedule experiments)

    /// A prompt split at the image/question boundary so [system + image] can be
    /// prefilled into a KV cache before the question exists.
    ///
    /// Guarantee: `full.text.tokens == prefix.text.tokens ++ suffixIDs` — the
    /// split point sits between the image-block special ids and the `")\n"`
    /// encode, so no tokenizer merge can cross it. A cache prefilled with
    /// `prefix` is therefore a valid prefix cache for `full`.
    public struct LabsParts {
        /// [system turn][<|im_start|>user\n(][image blocks] — pixels attached.
        public var prefix: LMInput
        /// [")\n"+question][<|im_end|>\n][<|im_start|>assistant…think block]
        public let suffixIDs: [Int]
        /// The whole sequence in one LMInput (for the cold/baseline arm).
        public var full: LMInput

        /// Drop the packed pixel strips once the encode is done.
        ///
        /// They feed visionTower.embeddings and nothing else — labsPrefill
        /// reads only input.text.tokens. Holding them through prefill and
        /// decode keeps a 9-strip image resident for the whole request for no
        /// reason. Releasing the last reference lets MLX reclaim it before
        /// prefill, which is the phase that now sets the peak.
        public mutating func releaseImagePixels() {
            prefix = LMInput(text: prefix.text, image: nil)
            full = LMInput(text: full.text, image: nil)
        }
    }

    /// Chat-template order is system → image → question; that order is what
    /// makes the prefix cache valid. `question` nil/empty = describe mode.
    public func labsPrepare(
        image: UserInput.Image,
        processing: UserInput.Processing? = nil,
        system: String,
        question: String?
    ) throws -> LabsParts {
        let (strips, frames, grid) = try processImageBlocks(image, processing: processing)
        let pixels = concatenated(strips, axis: 1).expandedDimensions(axis: 0)

        var prefixIds: [Int] = [imStartId]
        prefixIds += encode("system\n" + system)
        prefixIds += [imEndId]
        prefixIds += encode("\n")
        prefixIds += [imStartId]
        prefixIds += encode("user\n(")
        prefixIds += imageBlockIds(frames: frames, grid: grid)

        var suffixIds: [Int] = encode(")\n" + (question ?? ""))
        suffixIds += [imEndId]
        suffixIds += encode("\n")
        suffixIds += [imStartId]
        suffixIds += encode("assistant\n<think>\n\n</think>\n\n")

        func lmInput(_ ids: [Int], withImage: Bool) -> LMInput {
            LMInput(
                text: .init(tokens: MLXArray(ids.map { Int32($0) }).expandedDimensions(axis: 0)),
                image: withImage ? .init(pixels: pixels, frames: frames) : nil
            )
        }

        return LabsParts(
            prefix: lmInput(prefixIds, withImage: true),
            suffixIDs: suffixIds,
            full: lmInput(prefixIds + suffixIds, withImage: true)
        )
    }

    /// Suffix-only ids for asking another question of an already-prefilled
    /// prefix cache (same layout as `LabsParts.suffixIDs`).
    public func labsSuffixIDs(question: String?) -> [Int] {
        var ids: [Int] = encode(")\n" + (question ?? ""))
        ids += [imEndId]
        ids += encode("\n")
        ids += [imStartId]
        ids += encode("assistant\n<think>\n\n</think>\n\n")
        return ids
    }
}
