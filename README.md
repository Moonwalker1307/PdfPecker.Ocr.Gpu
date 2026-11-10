# PdfPecker.Ocr.Gpu

GPU support for [PdfPecker.Ocr](https://github.com/Moonwalker1307/PdfPecker.Ocr). It runs the text detection,
recognition and orientation networks on ONNX Runtime with CUDA (NVIDIA) or DirectML (any DirectX 12 GPU on Windows).
Any network you don't move to the GPU runs on PdfPecker.Ocr's own CPU runtime, and if there is no GPU everything falls
back to that runtime. Decoding, layout, hOCR and searchable PDFs are PdfPecker.Ocr's, unchanged.

PdfPecker.Ocr itself stays dependency-free. This package is where the native code lives.

> PdfPecker.Ocr.Gpu is being built towards its first release, 1.24.0, which shares PdfPecker's version number as
> PdfPecker.Ocr does. It needs PdfPecker.Ocr 1.24.0 or later, the first with `OcrOptions.Accelerator`. Until that is on
> nuget.org, it builds against a checkout of PdfPecker.Ocr next to this repository.

```csharp
using PdfPecker.Ocr;
using PdfPecker.Ocr.Gpu;

// CPU, GPU, or both: the detector on the GPU and the recognizer on the CPU
var options = new OcrOptions { Accelerator = GpuAccelerator.For(OcrDevice.Gpu) };
OcrDocument document = PdfOcr.Recognize(File.ReadAllBytes("scanned.pdf"), options);
```

## Install

```
dotnet add package PdfPecker.Ocr.Gpu
```

Then add the native ONNX Runtime for your hardware. The package only references ONNX Runtime's managed half, so the
application chooses:

| Hardware | Package | Also needs |
| --- | --- | --- |
| NVIDIA GPU, Windows or Linux | `Microsoft.ML.OnnxRuntime.Gpu` | CUDA 12 and cuDNN 9, as ONNX Runtime's [CUDA page](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) lists |
| Any DirectX 12 GPU, Windows | `Microsoft.ML.OnnxRuntime.DirectML` | A current GPU driver |
| No GPU, for testing | `Microsoft.ML.OnnxRuntime` | Nothing |

Reference only one of them, since they ship the same native library. The package targets `net8.0`, `net9.0` and
`net10.0`.

## Choosing the device

`GpuAccelerator.For(OcrDevice)` returns a shared accelerator for each mode:

- `OcrDevice.Cpu` returns null, which is plain PdfPecker.Ocr. ONNX Runtime isn't loaded.
- `OcrDevice.Gpu` runs all three networks on the GPU.
- `OcrDevice.Hybrid` runs the text detector on the GPU, because it reads the whole page at once and gains the most.
  The line recognizer and orientation classifier, which read small batches of lines, run on the CPU.

For more control, create your own accelerator and keep it for the life of the application. Its sessions hold the
model weights on the device, and the networks are pooled with PdfPecker.Ocr's workers for each accelerator instance.

```csharp
var gpu = new GpuAccelerator(new GpuOptions
{
    Provider = GpuProvider.Cuda,         // Auto (CUDA, then DirectML), Cuda, DirectML or OnnxRuntimeCpu
    DeviceId = 1,                        // the second GPU
    Networks = OcrNetworks.Detection | OcrNetworks.Recognition,
    FallBackToCpu = true,                // no GPU: run on PdfPecker.Ocr's CPU runtime instead of failing
});

Console.WriteLine(gpu.ActiveProvider);   // Cuda, or null when it fell back
Console.WriteLine(gpu.FallbackReason);   // why it fell back, such as a missing cuDNN
Console.WriteLine(string.Join(", ", GpuAccelerator.AvailableProviders()));
```

The provider is picked on first use: the first page, or the first read of `ActiveProvider`. Picking it loads the first
model onto the device, which proves that the provider is in the native build and that a device answers.

## Results

The networks are the same ONNX files PdfPecker.Ocr's own models are converted from, embedded unchanged. ONNX Runtime
and the GPU compute in their own order and precision, though, so positions and confidences can differ slightly from
the CPU runtime's, and very rarely a character can too. On the test pages, ONNX Runtime's CPU provider reads exactly
the same text as the built-in runtime, with line boxes within a pixel.

## Performance

`bench/PdfPecker.Ocr.Gpu.Benchmarks` times a synthetic 300 dpi A4 page of 40 lines, or the pages of a PDF you give it,
in each mode on one machine:

```
dotnet run -c Release --project bench/PdfPecker.Ocr.Gpu.Benchmarks -p:OnnxRuntimeNative=Gpu -- --runs 10
```

There are no GPU figures here yet. They come from a run on a real GPU, and the development machine so far had none.
On a 4-vCPU cloud VM without a GPU, the ONNX path on ONNX Runtime's CPU provider took 2.97 s a page (median), against
1.94 s for PdfPecker.Ocr's built-in runtime, with identical text. So the CPU provider is only for testing. A GPU only
helps if it beats the built-in runtime.

## Limitations

- **First call.** Loading the models onto a GPU and compiling its kernels takes seconds the first time. Keep the
  accelerator for the life of the application.
- **Copies.** Each network's input and output go through host memory. The recognizer returns the probability of each
  of its roughly 18,700 classes at every time step, which is the largest copy. Doing the argmax on the device is planned.
- **Cancellation** is checked between network runs, not during one.
- **Native AOT is untested.** PdfPecker.Ocr on its own is AOT-compatible. This package depends on ONNX Runtime's
  managed API, which hasn't been checked under Native AOT here.

## Build

```
git clone https://github.com/Moonwalker1307/PdfPecker.Ocr        # next to this repository, until 1.24.0 is released
dotnet build -c Release
dotnet test -c Release --no-build
```

The tests use the CPU build of ONNX Runtime. They run the ONNX path end to end, check its text against the built-in
runtime, check that CUDA and DirectML fall back when they can't start, and check that the embedded models match the
pinned hashes. The models are written by `tools/ModelPacker`, never by hand:

```
dotnet run -c Release --project tools/ModelPacker
```

## Licence

The code is MIT (`LICENSE`). The embedded models are Apache-2.0, copyright Baidu and/or the PaddleOCR rights holders,
converted to ONNX by the RapidOCR Authors. See `THIRD-PARTY-NOTICES.md`.
