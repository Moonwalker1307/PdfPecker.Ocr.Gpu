using System.Diagnostics;
using System.Globalization;
using PdfPecker;
using PdfPecker.Ocr;
using PdfPecker.Ocr.Gpu;

// Times one synthetic A4 page (or the pages of a PDF given with --pdf) on the CPU, the GPU and both, on this machine.
// Usage, from the repository root:
//   dotnet run -c Release --project bench/PdfPecker.Ocr.Gpu.Benchmarks -p:OnnxRuntimeNative=Gpu -- [--provider cuda]
//       [--runs 10] [--pdf file.pdf]
// OnnxRuntimeNative picks the native ONNX Runtime: Cpu (the default, any machine), Gpu (CUDA) or DirectML (Windows).
// The first run of each mode is reported apart, since it loads the models and, on a GPU, the kernels.

int runs = 10;
string? pdfPath = null;
var provider = GpuProvider.Auto;
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--runs" when i + 1 < args.Length: runs = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
        case "--pdf" when i + 1 < args.Length: pdfPath = args[++i]; break;
        case "--provider" when i + 1 < args.Length: provider = Enum.Parse<GpuProvider>(args[++i], ignoreCase: true); break;
        default: throw new ArgumentException($"Unknown argument {args[i]}.");
    }
}

byte[] pdf = pdfPath is null ? SyntheticPage() : File.ReadAllBytes(pdfPath);
Console.WriteLine($"ONNX Runtime providers: {string.Join(", ", GpuAccelerator.AvailableProviders())}");
Console.WriteLine($"{Environment.ProcessorCount} processors, {runs} runs per mode");
Console.WriteLine();
Console.WriteLine("| Mode | Runs on | First call | Median | Fastest | Characters |");
Console.WriteLine("|---|---|---:|---:|---:|---:|");

string? reference = null;
foreach (var (mode, accelerator) in new (string, GpuAccelerator?)[]
{
    ("CPU", null),
    ("GPU", new GpuAccelerator(new GpuOptions { Provider = provider })),
    ("Hybrid", new GpuAccelerator(new GpuOptions { Provider = provider, Networks = OcrNetworks.Detection })),
})
{
    var options = new OcrOptions { Accelerator = accelerator, PageMode = OcrPageMode.Force };
    string device = accelerator is null ? "built-in" : accelerator.ActiveProvider?.ToString() ?? "built-in (fallback)";
    var watch = Stopwatch.StartNew();
    OcrDocument document = PdfOcr.Recognize(pdf, options);
    double first = watch.Elapsed.TotalMilliseconds;
    var times = new List<double>();
    for (int r = 0; r < runs; r++)
    {
        watch.Restart();
        document = PdfOcr.Recognize(pdf, options);
        times.Add(watch.Elapsed.TotalMilliseconds);
    }

    times.Sort();
    reference ??= document.Text;
    string same = document.Text == reference ? "" : " (text differs from CPU)";
    Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
        $"| {mode} | {device} | {first:F0} ms | {times[times.Count / 2]:F0} ms | {times[0]:F0} ms | {document.Text.Length}{same} |"));
    if (accelerator?.FallbackReason is { } reason) Console.WriteLine($"  {mode} fell back: {reason}");
}

static byte[] SyntheticPage()
{
    var lines = new List<string>();
    for (int i = 1; i <= 40; i++)
    {
        lines.Add(string.Create(CultureInfo.InvariantCulture,
            $"Line {i}: The quick brown fox jumps over the lazy dog, invoice {2026_0000 + (i * 37)} totals NOK {i * 1234.5:N2}."));
    }

    string body = string.Concat(lines.Select(l => "<p>" + System.Net.WebUtility.HtmlEncode(l) + "</p>"));
    string html = "<!doctype html><html><head><meta charset=\"utf-8\"><style>@page{size:A4;margin:15mm}body{margin:0}"
        + "p{font:11pt/1.5 serif;margin:0}</style></head><body>" + body + "</body></html>";
    byte[] digital = HtmlToPdf.Convert(html);
    var image = PdfRenderer.Open(digital).RenderPage(0, new ImageOptions { Dpi = 300 });
    return ImageToPdf.Convert([image.ToPng()], null, new ImageToPdfOptions { FitPageToImage = true, Dpi = 300 });
}
