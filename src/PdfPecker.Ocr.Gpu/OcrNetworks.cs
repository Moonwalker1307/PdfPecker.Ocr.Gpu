namespace PdfPecker.Ocr.Gpu;

/// <summary>A set of PdfPecker.Ocr's networks, for choosing which of them run on the GPU.</summary>
[Flags]
public enum OcrNetworks
{
    /// <summary>None of them: everything runs on PdfPecker.Ocr's built-in CPU runtime.</summary>
    None = 0,

    /// <summary>The text detector, which reads the whole page at once and gains the most from a GPU.</summary>
    Detection = 1,

    /// <summary>The line recognizer, which reads the lines in batches.</summary>
    Recognition = 2,

    /// <summary>The small line orientation classifier.</summary>
    Orientation = 4,

    /// <summary>All three.</summary>
    All = Detection | Recognition | Orientation,
}
