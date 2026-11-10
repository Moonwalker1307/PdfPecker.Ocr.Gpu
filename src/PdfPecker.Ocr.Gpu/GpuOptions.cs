namespace PdfPecker.Ocr.Gpu;

/// <summary>Settings for <see cref="GpuAccelerator"/>. A fresh instance holds the defaults.</summary>
public sealed class GpuOptions
{
    /// <summary>The execution provider. Default: <see cref="GpuProvider.Auto"/>.</summary>
    public GpuProvider Provider { get; set; } = GpuProvider.Auto;

    /// <summary>The GPU to use, numbered from 0 as CUDA or DirectML number them. Default: 0.</summary>
    public int DeviceId { get; set; }

    /// <summary>
    /// The networks that run on the GPU; the others run on PdfPecker.Ocr's built-in CPU runtime. Default:
    /// <see cref="OcrNetworks.All"/>.
    /// </summary>
    public OcrNetworks Networks { get; set; } = OcrNetworks.All;

    /// <summary>
    /// When no provider can be started (no GPU, no driver, or an ONNX Runtime without that provider), run everything on
    /// PdfPecker.Ocr's built-in CPU runtime instead of failing. <see cref="GpuAccelerator.FallbackReason"/> says why.
    /// Default: true.
    /// </summary>
    public bool FallBackToCpu { get; set; } = true;

    /// <summary>Checks the settings, so a wrong value fails before any work starts.</summary>
    /// <exception cref="ArgumentOutOfRangeException">A setting is out of its range.</exception>
    internal void Validate()
    {
        if (!Enum.IsDefined(Provider))
            throw new ArgumentOutOfRangeException(nameof(Provider), Provider, "Provider is not a defined GpuProvider.");
        if (DeviceId < 0)
            throw new ArgumentOutOfRangeException(nameof(DeviceId), DeviceId, "DeviceId must be at least 0.");
        if ((Networks & ~OcrNetworks.All) != 0)
            throw new ArgumentOutOfRangeException(nameof(Networks), Networks, "Networks has flags that are not defined.");
    }
}
