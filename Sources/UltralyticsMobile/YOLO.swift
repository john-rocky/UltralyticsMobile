import Foundation
import UIKit
import SwiftUI

public class YOLO {
    var predictor: Predictor!
    var yoloView: YOLOView?
    
    public init(_ modelPathOrName: String, task: YOLOTask) {
        switch task {
        case .detect:
            predictor = ObjectDetector(modelPathOrName: modelPathOrName)
        }
    }
    
    public func callAsFunction(_ uiImage: UIImage, returnAnnotatedImage: Bool = true) -> YOLOResult {
        let ciImage = CIImage(image: uiImage)!
        var result = predictor.predictOnImage(image: ciImage)
        if returnAnnotatedImage {
            let annotatedImage = drawYOLODetections(on: ciImage, result: result)
            result.annotatedImage = annotatedImage
        }
        return result
    }
    
    public func callAsFunction(_ ciImage: CIImage, returnAnnotatedImage: Bool = true) -> YOLOResult {
        var result = predictor.predictOnImage(image: ciImage)
        if returnAnnotatedImage {
            let annotatedImage = drawYOLODetections(on: ciImage, result: result)
            result.annotatedImage = annotatedImage
        }
        return result
    }
    
    public func callAsFunction(_ cgImage: CGImage, returnAnnotatedImage: Bool = true) -> YOLOResult {
        let ciImage = CIImage(cgImage: cgImage)
        var result = predictor.predictOnImage(image: ciImage)
        if returnAnnotatedImage {
            let annotatedImage = drawYOLODetections(on: ciImage, result: result)
            result.annotatedImage = annotatedImage
        }
        return result
    }
    
    public func callAsFunction(
        _ resourceName: String,
        withExtension ext: String? = nil,
        returnAnnotatedImage: Bool = true
    ) -> YOLOResult {
        guard let url = Bundle.main.url(forResource: resourceName, withExtension: ext),
              let data = try? Data(contentsOf: url),
              let uiImage = UIImage(data: data)
        else {
            return YOLOResult(orig_shape: .zero, boxes: [])
        }
        return self(uiImage, returnAnnotatedImage: returnAnnotatedImage)
    }
    
    public func callAsFunction(
        _ remoteURL: URL?,
        returnAnnotatedImage: Bool = true
    ) -> YOLOResult {
        guard let remoteURL = remoteURL,
              let data = try? Data(contentsOf: remoteURL),
              let uiImage = UIImage(data: data)
        else {
            return YOLOResult(orig_shape: .zero, boxes: [])
        }
        return self(uiImage, returnAnnotatedImage: returnAnnotatedImage)
    }
    
    public func callAsFunction(
        _ localPath: String,
        returnAnnotatedImage: Bool = true
    ) -> YOLOResult {
        let fileURL = URL(fileURLWithPath: localPath)
        guard let data = try? Data(contentsOf: fileURL),
              let uiImage = UIImage(data: data)
        else {
            return YOLOResult(orig_shape: .zero, boxes: [])
        }
        return self(uiImage, returnAnnotatedImage: returnAnnotatedImage)
    }
    
    @MainActor @available(iOS 16.0, *)
    public func callAsFunction(
        _ swiftUIImage: SwiftUI.Image,
        returnAnnotatedImage: Bool = true
    ) -> YOLOResult {
        let renderer = ImageRenderer(content: swiftUIImage)
        guard let uiImage = renderer.uiImage else {
            return YOLOResult(orig_shape: .zero, boxes: [])
        }
        return self(uiImage, returnAnnotatedImage: returnAnnotatedImage)
    }
}