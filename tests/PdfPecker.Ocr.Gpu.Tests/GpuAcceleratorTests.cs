using System.Security.Cryptography;
using PdfPecker.Ocr.Gpu.Models;

namespace PdfPecker.Ocr.Gpu.Tests;

/// <summary>
/// The accelerator on the machines CI has, which have no GPU: the ONNX path on ONNX Runtime's CPU provider, and the
/// fallback to PdfPecker.Ocr's built-in runtime when CUDA or DirectML cannot start.
/// </summary>
public sealed class GpuAcceleratorTests
{
    private static readonly Lazy<OcrPage> s_builtIn = new(() => Read(null));

    [Fact]
    public void TheOnnxPathReadsTheSameTextAsTheBuiltInRuntime()
    {
        var accelerator = new GpuAccelerator(new GpuOptions { Provider = GpuProvider.OnnxRuntimeCpu });
        Assert.Equal(GpuProvider.OnnxRuntimeCpu, accelerator.ActiveProvider);
        Assert.Null(accelerator.FallbackReason);
        Assert.Equal("ONNX Runtime CPU", accelerator.Name);

        OcrPage page = Read(accelerator);
        Assert.Equal(s_builtIn.Value.Text, page.Text);
        Assert.Contains("Total NOK 17,589.45", page.Text, StringComparison.Ordinal);
        AssertClose(s_builtIn.Value, page);
    }

    [Fact]
    public void HybridRunsOnlyDetectionThroughOnnx()
    {
        var accelerator = new GpuAccelerator(new GpuOptions { Provider = GpuProvider.OnnxRuntimeCpu, Networks = OcrNetworks.Detection });
        using (IOcrNetwork? detection = accelerator.CreateNetwork(OcrNetworkKind.Detection))
        {
            Assert.NotNull(detection);
        }

        Assert.Null(accelerator.CreateNetwork(OcrNetworkKind.Recognition));
        Assert.Null(accelerator.CreateNetwork(OcrNetworkKind.Orientation));

        OcrPage page = Read(accelerator);
        Assert.Equal(s_builtIn.Value.Text, page.Text);
    }

    [Theory]
    [InlineData(GpuProvider.Auto)]
    [InlineData(GpuProvider.Cuda)]
    [InlineData(GpuProvider.DirectML)]
    public void WithoutAGpuItFallsBackToTheBuiltInRuntime(GpuProvider provider)
    {
        var accelerator = new GpuAccelerator(new GpuOptions { Provider = provider });
        Assert.Null(accelerator.ActiveProvider);
        Assert.Contains("could be started", accelerator.FallbackReason, StringComparison.Ordinal);
        Assert.Equal("CPU", accelerator.Name);
        foreach (OcrNetworkKind kind in Enum.GetValues<OcrNetworkKind>())
        {
            Assert.Null(accelerator.CreateNetwork(kind));
        }

        // Every network on the built-in runtime: the very same result as without an accelerator.
        OcrPage page = Read(accelerator);
        Assert.Equal(s_builtIn.Value.Text, page.Text);
        Assert.Equal(s_builtIn.Value.Lines.Select(l => (l.Quad, l.Confidence)), page.Lines.Select(l => (l.Quad, l.Confidence)));
    }

    [Fact]
    public void WithoutFallbackAMissingGpuFails()
    {
        var accelerator = new GpuAccelerator(new GpuOptions { Provider = GpuProvider.Cuda, FallBackToCpu = false });
        var error = Assert.Throws<InvalidOperationException>(() => Read(accelerator));
        Assert.Contains("Cuda", error.Message, StringComparison.Ordinal);
        Assert.Throws<InvalidOperationException>(() => accelerator.ActiveProvider);
    }

    [Fact]
    public void DevicesMapToSharedAccelerators()
    {
        Assert.Null(GpuAccelerator.For(OcrDevice.Cpu));
        Assert.Same(GpuAccelerator.For(OcrDevice.Gpu), GpuAccelerator.For(OcrDevice.Gpu));
        Assert.Equal(OcrNetworks.All, GpuAccelerator.For(OcrDevice.Gpu)!.Networks);
        Assert.Equal(OcrNetworks.Detection, GpuAccelerator.For(OcrDevice.Hybrid)!.Networks);
        Assert.Throws<ArgumentOutOfRangeException>(() => GpuAccelerator.For((OcrDevice)7));
    }

    [Fact]
    public void TheCpuBuildOfOnnxRuntimeListsOnlyTheCpuProvider()
    {
        IReadOnlyList<string> providers = GpuAccelerator.AvailableProviders();
        Assert.Contains("CPUExecutionProvider", providers);
        Assert.DoesNotContain("CUDAExecutionProvider", providers);
    }

    [Fact]
    public void OptionsAreChecked()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new GpuAccelerator(new GpuOptions { DeviceId = -1 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new GpuAccelerator(new GpuOptions { Provider = (GpuProvider)9 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new GpuAccelerator(new GpuOptions { Networks = (OcrNetworks)8 }));
    }

    [Theory]
    [InlineData(OcrNetworkKind.Detection, "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f")]
    [InlineData(OcrNetworkKind.Recognition, "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884")]
    [InlineData(OcrNetworkKind.Orientation, "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c")]
    public void TheEmbeddedModelsAreTheRapidocrFilesUnchanged(OcrNetworkKind kind, string sha256)
    {
        Assert.Equal(sha256, Convert.ToHexString(SHA256.HashData(ModelData.Load(kind))).ToLowerInvariant());
    }

    private static OcrPage Read(OcrAccelerator? accelerator)
    {
        var (rgba, width, height) = TestPages.Invoice;
        return ImageOcr.RecognizeRgba(rgba, width, height, new OcrOptions { Accelerator = accelerator });
    }

    /// <summary>The same lines, with boxes within a pixel and confidences within a hundredth.</summary>
    private static void AssertClose(OcrPage expected, OcrPage actual)
    {
        Assert.Equal(expected.Lines.Count, actual.Lines.Count);
        for (int i = 0; i < expected.Lines.Count; i++)
        {
            OcrBox a = expected.Lines[i].Box, b = actual.Lines[i].Box;
            Assert.InRange(Math.Abs(a.X - b.X), 0, 1);
            Assert.InRange(Math.Abs(a.Y - b.Y), 0, 1);
            Assert.InRange(Math.Abs(a.Width - b.Width), 0, 1);
            Assert.InRange(Math.Abs(a.Height - b.Height), 0, 1);
            Assert.InRange(Math.Abs(expected.Lines[i].Confidence - actual.Lines[i].Confidence), 0, 0.01);
        }
    }
}
