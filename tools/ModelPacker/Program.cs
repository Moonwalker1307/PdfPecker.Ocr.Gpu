using System.IO.Compression;
using System.Security.Cryptography;

// Writes src/PdfPecker.Ocr.Gpu/Models/{det,rec,cls}.onnx.deflate: the ONNX models of the rapidocr 3.10.0 wheel on PyPI,
// unchanged and deflated. They are the files PdfPecker.Ocr's ModelBuilder converts for its own runtime, pinned by the
// same hashes, so both packages run the same networks. Usage, from the repository root:
//   dotnet run -c Release --project tools/ModelPacker -- [--wheel path] [--out dir]
// Without --wheel the wheel is downloaded once into a cache folder in the temporary directory. The output is
// byte-identical on every run.

const string WheelUrl = "https://files.pythonhosted.org/packages/59/07/203065283bba0c5e9c10af3602dc25ab3e59ee54dc2d1d22c8faeabf5ed8/rapidocr-3.10.0-py3-none-any.whl";
const string WheelSha256 = "2fc34e26cd0f48514804a94f8832283039c7334076bff60768b1fed9ad068050";

var models = new (string Name, string Entry, string Sha256)[]
{
    ("det", "rapidocr/models/PP-OCRv6_det_small.onnx", "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f"),
    ("rec", "rapidocr/models/PP-OCRv6_rec_small.onnx", "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884"),
    ("cls", "rapidocr/models/ch_ppocr_mobile_v2.0_cls_mobile.onnx", "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c"),
};

string? wheel = null;
string output = Path.Combine("src", "PdfPecker.Ocr.Gpu", "Models");
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--wheel" when i + 1 < args.Length: wheel = args[++i]; break;
        case "--out" when i + 1 < args.Length: output = args[++i]; break;
        default: throw new ArgumentException($"Unknown argument {args[i]}.");
    }
}

wheel ??= await Download();
string wheelHash = Sha256(File.ReadAllBytes(wheel));
if (wheelHash != WheelSha256) throw new InvalidDataException($"{wheel} has SHA-256 {wheelHash}, expected {WheelSha256}.");

Directory.CreateDirectory(output);
using ZipArchive archive = ZipFile.OpenRead(wheel);
foreach (var (name, entry, sha256) in models)
{
    ZipArchiveEntry file = archive.GetEntry(entry) ?? throw new InvalidDataException($"{entry} is missing from the wheel.");
    using var data = new MemoryStream();
    using (Stream stream = file.Open()) stream.CopyTo(data);
    string hash = Sha256(data.ToArray());
    if (hash != sha256) throw new InvalidDataException($"{entry} has SHA-256 {hash}, expected {sha256}.");

    using var packed = new MemoryStream();
    using (var deflate = new DeflateStream(packed, CompressionLevel.SmallestSize, leaveOpen: true)) deflate.Write(data.GetBuffer(), 0, (int)data.Length);
    string target = Path.Combine(output, name + ".onnx.deflate");
    File.WriteAllBytes(target, packed.ToArray());
    Console.WriteLine($"  written {target}: {data.Length:N0} bytes, {packed.Length:N0} compressed (model SHA-256 {sha256})");
}

static async Task<string> Download()
{
    string cache = Path.Combine(Path.GetTempPath(), "pdfpecker-ocr-gpu-models");
    Directory.CreateDirectory(cache);
    string path = Path.Combine(cache, "rapidocr-3.10.0-py3-none-any.whl");
    if (!File.Exists(path))
    {
        using var http = new HttpClient();
        byte[] data = await http.GetByteArrayAsync(new Uri(WheelUrl));
        await File.WriteAllBytesAsync(path, data);
    }

    return path;
}

static string Sha256(byte[] data) => Convert.ToHexString(SHA256.HashData(data)).ToLowerInvariant();
