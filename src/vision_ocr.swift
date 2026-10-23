// Apple Vision OCR -> JSON on stdout. No install: uses the OS Vision framework.
//   swiftc -O ocr_bench/vision_ocr.swift -o <bin>; <bin> page.png [ja-JP,en-US | auto]
// Output: {"width":W,"height":H,"lines":[{"text","conf","bbox":[x0,y0,x1,y1],"chars":[{"c","bbox"}]}]}
// Boxes are pixels, origin top-left (Vision's normalized lower-left origin is flipped here).
// Language correction is off: it can "correct" digits.
import Foundation
import Vision
import AppKit

let path = CommandLine.arguments[1]
guard let img = NSImage(contentsOfFile: path),
      let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    FileHandle.standardError.write("cannot read \(path)\n".data(using: .utf8)!)
    exit(1)
}
let W = Double(cg.width), H = Double(cg.height)

func px(_ r: CGRect) -> [Double] {
    [r.minX * W, (1 - r.maxY) * H, r.maxX * W, (1 - r.minY) * H].map { ($0 * 10).rounded() / 10 }
}

let req = VNRecognizeTextRequest()
req.recognitionLevel = .accurate
let langs = CommandLine.arguments.count > 2 ? CommandLine.arguments[2] : "ja-JP,en-US"
if langs == "auto" {
    req.automaticallyDetectsLanguage = true
} else {
    req.recognitionLanguages = langs.split(separator: ",").map(String.init)
}
req.usesLanguageCorrection = false
try VNImageRequestHandler(cgImage: cg, options: [:]).perform([req])

var lines: [[String: Any]] = []
for obs in req.results ?? [] {
    guard let cand = obs.topCandidates(1).first else { continue }
    let s = cand.string
    var chars: [[String: Any]] = []
    var i = s.startIndex
    while i < s.endIndex {
        let j = s.index(after: i)
        let ch = String(s[i..<j])
        if ch.trimmingCharacters(in: .whitespaces).isEmpty {
            chars.append(["c": " ", "bbox": NSNull()])  // keep spaces: they separate numbers
        } else if let box = try? cand.boundingBox(for: i..<j) {
            chars.append(["c": ch, "bbox": px(box.boundingBox)])
        }
        i = j
    }
    lines.append(["text": s, "conf": Double(cand.confidence), "bbox": px(obs.boundingBox), "chars": chars])
}
let out: [String: Any] = ["width": W, "height": H, "lines": lines]
FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject: out))
