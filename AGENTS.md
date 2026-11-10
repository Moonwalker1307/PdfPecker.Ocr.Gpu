# AGENTS.md

Guidance for coding agents (and people) working in this repository. `CLAUDE.md` imports this file.

PdfPecker.Ocr.Gpu runs PdfPecker.Ocr's three networks (text detection, line recognition, line orientation) on ONNX
Runtime with CUDA or DirectML, behind PdfPecker.Ocr's public accelerator seam (`OcrAccelerator`, `IOcrNetwork`,
`OcrNetworkKind`). It exists so that PdfPecker.Ocr can stay free of native code: everything that needs a GPU, a native
library or ONNX Runtime belongs here, and nothing here may require changes to PdfPecker.Ocr beyond its public API.
PdfPecker.Ocr's `AGENTS.md` is the source of the shared rules; this file says where this repository differs.

## Core rules

- **Dependencies.** `src/PdfPecker.Ocr.Gpu` references PdfPecker.Ocr and `Microsoft.ML.OnnxRuntime.Managed` at
  `$(OnnxRuntimeManagedVersion)` (the floor; the application's native package raises it), and nothing else. Never
  reference a native ONNX Runtime package from the library: the application picks CUDA, DirectML or CPU. Tests and the
  benchmark reference the CPU build (`$(OnnxRuntimeTestVersion)`).
- **PdfPecker.Ocr.** Until a release with `OcrOptions.Accelerator` is on nuget.org, the projects build against a
  checkout of PdfPecker.Ocr next to this repository (`../PdfPecker.Ocr`, see `Directory.Build.props`), and CI checks
  out the ref in `PDFPECKER_OCR_REF`. When the seam needs something new, change PdfPecker.Ocr in its own pull request.
- **Models are generated data.** `src/PdfPecker.Ocr.Gpu/Models/*.onnx.deflate` are written by `tools/ModelPacker` from
  the rapidocr 3.10.0 wheel, pinned by the same SHA-256 hashes as PdfPecker.Ocr's ModelBuilder, and are the ONNX files
  unchanged. Never hand-edit them; a new model or source updates `THIRD-PARTY-NOTICES.md` and the hash tests together.
- **Fallback, not failure.** With `FallBackToCpu` (the default) a missing GPU, driver or provider must end in
  PdfPecker.Ocr's built-in runtime, never an exception. Only catch the exceptions that mean "cannot start here"
  (`OnnxRuntimeException` and native load failures); anything else is a bug and must reach the caller.
- **The seam's contract.** A network's output stays valid until its next run, and a network is used by one thread at a
  time; sessions are shared between workers, since `InferenceSession.Run` is thread-safe.
- **Licences.** The code is MIT; the package's licence expression is `MIT AND Apache-2.0` and it carries
  `THIRD-PARTY-NOTICES.md`. Take nothing copyleft.
- **Claims.** Never put GPU speed figures in the README that were not measured with `bench/PdfPecker.Ocr.Gpu.Benchmarks`
  on a named machine. Cloud VMs used for development have no GPU.

## Build and verify

```
dotnet build -c Release
dotnet test -c Release --no-build
dotnet run -c Release --project bench/PdfPecker.Ocr.Gpu.Benchmarks -- --runs 5   # -p:OnnxRuntimeNative=Gpu on a GPU
```

The build must be warning-free (warnings are errors) and the tests must pass on .NET 8, 9 and 10. The tests cannot see
a GPU on CI: a change to the CUDA or DirectML paths needs a run on a machine with one, noted in the pull request.

## Commit guidance

As in PdfPecker.Ocr: a branch per piece of work from the latest `origin/main`, named after the work with a `claude/`
prefix when an agent creates it, short descriptive commit subjects, and a body that says why.
