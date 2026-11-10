namespace PdfPecker.Ocr.Gpu;

/// <summary>Where OCR runs, in the terms of <see cref="GpuAccelerator.For(OcrDevice)"/>.</summary>
public enum OcrDevice
{
    /// <summary>On the CPU, with PdfPecker.Ocr's built-in runtime; no ONNX Runtime is loaded.</summary>
    Cpu,

    /// <summary>Every network on the GPU, falling back to the CPU when there is none.</summary>
    Gpu,

    /// <summary>
    /// Both: the text detector, which reads the whole page at once and gains the most from a GPU, on the GPU, and the
    /// line recognizer and orientation classifier, which read small batches of lines, on the CPU's built-in runtime.
    /// Suits a small GPU, a GPU that other work shares, or a strong CPU; falls back to the CPU when there is no GPU.
    /// </summary>
    Hybrid,
}
