using System.IO.Compression;

namespace PdfPecker.Ocr.Gpu.Models;

/// <summary>
/// The ONNX models, embedded deflated by tools/ModelPacker: the files of the rapidocr 3.10.0 wheel that PdfPecker.Ocr's
/// own models are converted from, unchanged. See THIRD-PARTY-NOTICES.md.
/// </summary>
internal static class ModelData
{
    /// <summary>The short name of the model of a network.</summary>
    public static string NameOf(OcrNetworkKind kind) => kind switch
    {
        OcrNetworkKind.Detection => "det",
        OcrNetworkKind.Recognition => "rec",
        OcrNetworkKind.Orientation => "cls",
        _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, "Not a defined OcrNetworkKind."),
    };

    /// <summary>The ONNX file of a network's model, inflated.</summary>
    public static byte[] Load(OcrNetworkKind kind)
    {
        string resource = "PdfPecker.Ocr.Gpu.Models." + NameOf(kind) + ".onnx.deflate";
        var assembly = typeof(ModelData).Assembly;
        using Stream stream = assembly.GetManifestResourceStream(resource)
            ?? throw new InvalidOperationException($"The model resource {resource} is missing from {assembly.FullName}.");
        using var deflate = new DeflateStream(stream, CompressionMode.Decompress);
        // The files inflate to at most a third more than their size, so the buffer is allocated once.
        using var data = new MemoryStream(capacity: (int)Math.Min(stream.Length * 4 / 3, int.MaxValue));
        deflate.CopyTo(data);
        return data.ToArray();
    }
}
