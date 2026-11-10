namespace PdfPecker.Ocr.Gpu;

/// <summary>Which ONNX Runtime execution provider runs the networks.</summary>
public enum GpuProvider
{
    /// <summary>CUDA if the application's ONNX Runtime has it and a device answers, then DirectML, then none.</summary>
    Auto,

    /// <summary>NVIDIA GPUs through CUDA and cuDNN, from the Microsoft.ML.OnnxRuntime.Gpu package, on Windows and Linux.</summary>
    Cuda,

    /// <summary>Any DirectX 12 GPU (NVIDIA, AMD, Intel) through DirectML, from the Microsoft.ML.OnnxRuntime.DirectML package, on Windows.</summary>
    DirectML,

    /// <summary>
    /// ONNX Runtime's own CPU provider. It is slower than PdfPecker.Ocr's built-in runtime and is there to test the
    /// ONNX path on machines without a GPU.
    /// </summary>
    OnnxRuntimeCpu,
}
