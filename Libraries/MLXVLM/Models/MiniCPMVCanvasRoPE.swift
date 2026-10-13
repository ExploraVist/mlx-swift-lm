//
//  MiniCPMVCanvasRoPE.swift
//  mlx-swift-lm
//
//  "Canvas" M-RoPE position ids for MiniCPM-V 4.7 (`minicpmv4_7`).
//
//  Port of `CanvasRoPE.get_rope_index` in mlx-vlm `models/minicpmv4_7/rope.py`, itself
//  identical to transformers `MiniCPMV4_7Model._group_visual_frames` /
//  `get_vision_position_ids` / `get_rope_index`. Images only (the Swift processor has no
//  video path), one sequence, no padding.
//
//  An image's thumbnail and slices are laid out on one 2-D canvas: every visual token
//  shares the temporal position of the canvas origin, its (h, w) positions are spread
//  over the canvas, and the surrounding markers (<image>, </image>, <slice>, </slice>,
//  the row separator "\n") are pinned to the canvas corners and edges. Text after the
//  image resumes at `origin + max(canvasH, canvasW) + 1`.
//

import Foundation

public enum MiniCPMVCanvasRoPE {

    /// Token ids the canvas layout keys on.
    public struct Markers: Sendable, Equatable {
        public var imageTokenId: Int
        public var imageStartId: Int
        public var imageEndId: Int
        public var sliceStartId: Int
        public var sliceEndId: Int
        public var newlineId: Int

        public init(
            imageTokenId: Int, imageStartId: Int, imageEndId: Int, sliceStartId: Int,
            sliceEndId: Int, newlineId: Int
        ) {
            self.imageTokenId = imageTokenId
            self.imageStartId = imageStartId
            self.imageEndId = imageEndId
            self.sliceStartId = sliceStartId
            self.sliceEndId = sliceEndId
            self.newlineId = newlineId
        }

        /// Ids of the MiniCPM-V 4.7 vocabulary (as stored in its `config.json`).
        public static let minicpmV47 = Markers(
            imageTokenId: 248_056, imageStartId: 248_078, imageEndId: 248_079,
            sliceStartId: 248_088, sliceEndId: 248_089, newlineId: 198)

        fileprivate var frameMarkers: Set<Int> {
            [sliceStartId, sliceEndId, imageStartId, imageEndId, newlineId]
        }
    }

    public enum Error: LocalizedError {
        case gridMismatch(String)

        public var errorDescription: String? {
            switch self {
            case .gridMismatch(let message): return "MiniCPM-V canvas RoPE: \(message)"
            }
        }
    }

    private struct Frame {
        var thumbStart: Int
        var thumbEnd: Int
        var slices: [(start: Int, end: Int)] = []
        var end: Int = 0
    }

    private struct Group {
        var start: Int
        var end: Int = 0
        var frames: [Frame]
    }

    /// Computes the [3][L] (temporal, height, width) position ids of `inputIds` and the
    /// rope delta (`max(position) + 1 - L`); decode positions are `cacheOffset + delta`.
    ///
    /// - Parameters:
    ///   - grids: (h, w) patch grid of every visual span in order (thumbnail, then its
    ///     slices row-major), i.e. the processor's `THW` frames / reference `target_sizes`.
    ///   - mergeFactor: patch-to-token factor per side: 4 for 16x downsampling, 2 for 4x.
    public static func positionIds(
        inputIds: [Int], grids: [(h: Int, w: Int)], markers: Markers, mergeFactor: Int = 4
    ) throws -> (positions: [[Int]], ropeDelta: Int) {
        let count = inputIds.count
        var t = Array(0 ..< count)
        var h = t
        var w = t

        func fill(_ range: Range<Int>, from value: Int) {
            for (k, i) in range.enumerated() {
                t[i] = value + k
                h[i] = value + k
                w[i] = value + k
            }
        }

        var gridIterator = grids.makeIterator()
        func nextGrid() throws -> (h: Int, w: Int) {
            guard let grid = gridIterator.next() else {
                throw Error.gridMismatch("more visual spans than image grids")
            }
            return grid
        }
        func checkSpan(_ span: Range<Int>, _ grid: (h: Int, w: Int)) throws {
            guard grid.h > 0, grid.w > 0, grid.h % mergeFactor == 0, grid.w % mergeFactor == 0,
                (grid.h / mergeFactor) * (grid.w / mergeFactor) == span.count
            else {
                throw Error.gridMismatch(
                    "grid \(grid.h)x\(grid.w) does not match a span of \(span.count) tokens")
            }
        }

        var currentPos = 0
        var currentIdx = 0
        for group in groupVisualFrames(inputIds, markers: markers) {
            if group.start > currentIdx {
                fill(currentIdx ..< group.start, from: currentPos)
                currentPos += group.start - currentIdx
            }
            var frameCursor = group.start
            for frame in group.frames {
                let thumbGrid = try nextGrid()
                try checkSpan(frame.thumbStart ..< frame.thumbEnd, thumbGrid)
                let frameStart = frame.thumbStart - 1
                if frameStart > frameCursor {
                    fill(frameCursor ..< frameStart, from: currentPos)
                    currentPos += frameStart - frameCursor + 1
                }
                let frameEnd = min(frame.end, group.end)
                let origin = currentPos
                let halo = max(origin - 1, 0)

                var canvasH = thumbGrid.h / mergeFactor
                var canvasW = thumbGrid.w / mergeFactor
                var numCols = 0
                var sliceH = 0
                var sliceW = 0
                var firstSliceGrid: (h: Int, w: Int) = (0, 0)
                let slices = frame.slices
                if !slices.isEmpty {
                    numCols = slices.count
                    for k in 0 ..< (slices.count - 1)
                    where slices[k + 1].start - slices[k].end > 2 {
                        numCols = k + 1
                        break
                    }
                    var numRows = slices.count / numCols
                    if numRows * numCols != slices.count {
                        numRows = 1
                        numCols = slices.count
                    }
                    firstSliceGrid = try nextGrid()
                    sliceH = firstSliceGrid.h / mergeFactor
                    sliceW = firstSliceGrid.w / mergeFactor
                    canvasH = numRows * sliceH
                    canvasW = numCols * sliceW
                }

                for i in max(frameStart, 0) ..< frameEnd {
                    t[i] = origin
                    h[i] = origin
                    w[i] = origin
                }
                if frameStart >= 0 {
                    h[frameStart] = halo
                    w[frameStart] = halo
                }
                if frame.thumbEnd < frameEnd {
                    h[frame.thumbEnd] = origin + canvasH
                    w[frame.thumbEnd] = origin + canvasW
                }
                place(
                    frame.thumbStart, grid: thumbGrid, origin: origin, canvasH: canvasH,
                    canvasW: canvasW, hOffset: 0, wOffset: 0, mergeFactor: mergeFactor,
                    t: &t, h: &h, w: &w)

                for (k, slice) in slices.enumerated() {
                    let hOffset = k / numCols * sliceH
                    let wOffset = k % numCols * sliceW
                    let grid = k == 0 ? firstSliceGrid : try nextGrid()
                    try checkSpan(slice.start ..< slice.end, grid)
                    let gridH = grid.h / mergeFactor
                    let gridW = grid.w / mergeFactor
                    let marker = slice.start - 1
                    if marker >= frameStart {
                        h[marker] = origin + hOffset
                        w[marker] = origin + wOffset
                    }
                    if slice.end < frameEnd {
                        h[slice.end] = origin + hOffset + gridH - 1
                        w[slice.end] = origin + wOffset + gridW - 1
                    }
                    place(
                        slice.start, grid: grid, origin: origin, canvasH: 0, canvasW: 0,
                        hOffset: hOffset, wOffset: wOffset, mergeFactor: mergeFactor,
                        t: &t, h: &h, w: &w)
                }
                // Row separators between slice rows sit just past the right edge.
                if slices.count > 1 {
                    for k in 0 ..< (slices.count - 1) {
                        let gapStart = slices[k].end
                        let gapEnd = slices[k + 1].start
                        if gapEnd - gapStart <= 2 { continue }
                        let boundaryH = (k / numCols + 1) * sliceH - 1
                        let rightEdgeW = numCols * sliceW
                        for i in (gapStart + 1) ..< (gapEnd - 1) {
                            h[i] = origin + boundaryH
                            w[i] = origin + rightEdgeW
                        }
                    }
                }
                currentPos = origin + max(canvasH, canvasW) + 1
                frameCursor = frameEnd
            }
            if frameCursor < group.end {
                fill(frameCursor ..< group.end, from: currentPos)
                currentPos += group.end - frameCursor
            }
            currentIdx = group.end
        }
        if currentIdx < count {
            fill(currentIdx ..< count, from: currentPos)
        }
        if gridIterator.next() != nil {
            throw Error.gridMismatch("more image grids than visual spans")
        }

        let maxPosition = max(t.max() ?? -1, h.max() ?? -1, w.max() ?? -1)
        return ([t, h, w], maxPosition + 1 - count)
    }

    /// Positions of one visual span: `linspace(0, canvas - 1, tokens)` per axis, rounded
    /// half-to-even like `np.rint`, offset into the canvas.
    private static func place(
        _ start: Int, grid: (h: Int, w: Int), origin: Int, canvasH: Int, canvasW: Int,
        hOffset: Int, wOffset: Int, mergeFactor: Int,
        t: inout [Int], h: inout [Int], w: inout [Int]
    ) {
        let gridH = grid.h / mergeFactor
        let gridW = grid.w / mergeFactor
        let hCoords = linspaceRounded(stop: (canvasH > 0 ? canvasH : gridH) - 1, count: gridH)
        let wCoords = linspaceRounded(stop: (canvasW > 0 ? canvasW : gridW) - 1, count: gridW)
        var i = start
        for y in hCoords {
            for x in wCoords {
                t[i] = origin
                h[i] = origin + hOffset + y
                w[i] = origin + wOffset + x
                i += 1
            }
        }
    }

    /// `np.rint(np.linspace(0, stop, count))`, mirroring numpy's arithmetic
    /// (`i * (stop / (count - 1))`, last element exactly `stop`).
    static func linspaceRounded(stop: Int, count: Int) -> [Int] {
        guard count > 1 else { return count == 1 ? [0] : [] }
        let step = Double(stop) / Double(count - 1)
        return (0 ..< count).map { i in
            let value = i == count - 1 ? Double(stop) : Double(i) * step
            return Int(value.rounded(.toNearestOrEven))
        }
    }

    /// `_group_visual_frames`: visual spans become frames (a thumbnail plus the slices that
    /// directly follow it); frames separated only by marker tokens share a group.
    private static func groupVisualFrames(_ ids: [Int], markers: Markers) -> [Group] {
        let markerSet = markers.frameMarkers
        var groups: [Group] = []
        var hasOpenFrame = false
        var previousEnd = 0

        var i = 0
        while i < ids.count {
            guard ids[i] == markers.imageTokenId else {
                i += 1
                continue
            }
            let start = i
            while i < ids.count, ids[i] == markers.imageTokenId { i += 1 }
            let end = i

            if hasOpenFrame, start > 0, ids[start - 1] == markers.sliceStartId {
                groups[groups.count - 1].frames[groups[groups.count - 1].frames.count - 1]
                    .slices.append((start, end))
            } else {
                let frame = Frame(thumbStart: start, thumbEnd: end)
                let gapEnd = max(start - 1, previousEnd)
                let adjacent =
                    !groups.isEmpty && ids[previousEnd ..< gapEnd].allSatisfy(markerSet.contains)
                if adjacent {
                    groups[groups.count - 1].frames.append(frame)
                } else {
                    groups.append(Group(start: start - 1, frames: [frame]))
                }
                hasOpenFrame = true
            }
            previousEnd = end
        }

        for g in groups.indices {
            for f in groups[g].frames.indices {
                let frame = groups[g].frames[f]
                var end = frame.slices.last?.end ?? frame.thumbEnd
                while end < ids.count, markerSet.contains(ids[end]) { end += 1 }
                groups[g].frames[f].end = end
            }
            groups[g].end = groups[g].frames.last!.end
        }
        return groups
    }
}
