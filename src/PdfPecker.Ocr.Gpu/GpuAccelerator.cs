using Microsoft.ML.OnnxRuntime;
using PdfPecker.Ocr.Gpu.Models;

namespace PdfPecker.Ocr.Gpu;

/// <summary>
/// Runs PdfPecker.Ocr's networks on a GPU through ONNX Runtime. Set it as <see cref="OcrOptions.Accelerator"/>, or take
/// a shared one from <see cref="For(OcrDevice)"/>:
/// <code>
/// var options = new OcrOptions { Accelerator = GpuAccelerator.For(OcrDevice.Gpu) };
/// OcrDocument document = PdfOcr.Recognize(File.ReadAllBytes("scanned.pdf"), options);
/// </code>
/// </summary>
/// <remarks>
/// <para>
/// The package references only ONNX Runtime's managed half. The application adds the native build for its hardware:
/// Microsoft.ML.OnnxRuntime.Gpu for NVIDIA GPUs (CUDA 12 and cuDNN 9 must be installed), Microsoft.ML.OnnxRuntime.DirectML
/// for any DirectX 12 GPU on Windows, or Microsoft.ML.OnnxRuntime to test on the CPU.
/// </para>
/// <para>
/// The provider is started on first use: the first page, or the first read of <see cref="ActiveProvider"/>,
/// <see cref="Name"/> or <see cref="FallbackReason"/>. If none starts and <see cref="GpuOptions.FallBackToCpu"/> is set,
/// every network runs on PdfPecker.Ocr's built-in runtime, as without an accelerator. The sessions, one per network
/// with its weights on the device, are loaded when a page first needs them and kept for the accelerator's lifetime,
/// so create one accelerator and keep it, or use <see cref="For(OcrDevice)"/>.
/// </para>
/// <para>
/// A GPU computes in its own order and precision, so text, positions and confidences can differ slightly from the
/// built-in runtime's.
/// </para>
/// </remarks>
public sealed class GpuAccelerator : OcrAccelerator
{
    private static readonly Lazy<GpuAccelerator> s_gpu = new(() => new GpuAccelerator(new GpuOptions()));
    private static readonly Lazy<GpuAccelerator> s_hybrid = new(() => new GpuAccelerator(new GpuOptions { Networks = OcrNetworks.Detection }));

    private readonly GpuProvider _provider;
    private readonly int _deviceId;
    private readonly OcrNetworks _networks;
    private readonly bool _fallBackToCpu;
    private readonly object _lock = new();
    private readonly InferenceSession?[] _sessions = new InferenceSession?[3];
    private bool _started;
    private SessionOptions? _sessionOptions;
    private GpuProvider? _active;
    private string? _fallbackReason;

    /// <summary>An accelerator with the given settings, which are copied.</summary>
    /// <param name="options">The provider, device and networks; null for the defaults.</param>
    /// <exception cref="ArgumentOutOfRangeException">A setting is out of its range.</exception>
    public GpuAccelerator(GpuOptions? options = null)
    {
        options ??= new GpuOptions();
        options.Validate();
        _provider = options.Provider;
        _deviceId = options.DeviceId;
        _networks = options.Networks;
        _fallBackToCpu = options.FallBackToCpu;
    }

    /// <summary>
    /// The shared accelerator for a device with the default provider and fallback: null for <see cref="OcrDevice.Cpu"/>,
    /// which is PdfPecker.Ocr without an accelerator, and one instance each for <see cref="OcrDevice.Gpu"/> and
    /// <see cref="OcrDevice.Hybrid"/>.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="device"/> is not a defined <see cref="OcrDevice"/>.</exception>
    public static GpuAccelerator? For(OcrDevice device) => device switch
    {
        OcrDevice.Cpu => null,
        OcrDevice.Gpu => s_gpu.Value,
        OcrDevice.Hybrid => s_hybrid.Value,
        _ => throw new ArgumentOutOfRangeException(nameof(device), device, "Not a defined OcrDevice."),
    };

    /// <summary>
    /// The execution providers the application's ONNX Runtime was built with, by ONNX Runtime's names (such as
    /// "CUDAExecutionProvider" or "DmlExecutionProvider"); empty when no native ONNX Runtime can be loaded. Having a
    /// provider does not mean a device answers; <see cref="ActiveProvider"/> says whether one did.
    /// </summary>
    public static IReadOnlyList<string> AvailableProviders()
    {
        try
        {
            return OrtEnv.Instance().GetAvailableProviders();
        }
        catch (Exception e) when (IsLoadFailure(e))
        {
            return [];
        }
    }

    /// <summary>The provider that runs the networks, or null when they run on the built-in CPU runtime.</summary>
    /// <exception cref="InvalidOperationException">No provider started and <see cref="GpuOptions.FallBackToCpu"/> is off.</exception>
    public GpuProvider? ActiveProvider
    {
        get
        {
            Start();
            return _active;
        }
    }

    /// <summary>Why the networks run on the built-in CPU runtime instead of a provider, or null when a provider runs them.</summary>
    /// <exception cref="InvalidOperationException">No provider started and <see cref="GpuOptions.FallBackToCpu"/> is off.</exception>
    public string? FallbackReason
    {
        get
        {
            Start();
            return _fallbackReason;
        }
    }

    /// <summary>The networks this accelerator runs on the provider; the others run on the built-in CPU runtime.</summary>
    public OcrNetworks Networks => _networks;

    /// <inheritdoc/>
    /// <exception cref="InvalidOperationException">No provider started and <see cref="GpuOptions.FallBackToCpu"/> is off.</exception>
    public override string Name => ActiveProvider switch
    {
        GpuProvider.Cuda => "CUDA",
        GpuProvider.DirectML => "DirectML",
        GpuProvider.OnnxRuntimeCpu => "ONNX Runtime CPU",
        _ => "CPU",
    };

    /// <inheritdoc/>
    /// <exception cref="InvalidOperationException">No provider started and <see cref="GpuOptions.FallBackToCpu"/> is off.</exception>
    public override IOcrNetwork? CreateNetwork(OcrNetworkKind kind)
    {
        if ((_networks & FlagOf(kind)) == 0)
        {
            return null;
        }

        Start();
        if (_active is null)
        {
            return null;
        }

        return new OnnxNetwork(Session(kind));
    }

    private static OcrNetworks FlagOf(OcrNetworkKind kind) => kind switch
    {
        OcrNetworkKind.Detection => OcrNetworks.Detection,
        OcrNetworkKind.Recognition => OcrNetworks.Recognition,
        OcrNetworkKind.Orientation => OcrNetworks.Orientation,
        _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, "Not a defined OcrNetworkKind."),
    };

    /// <summary>The session of a network, loaded on first use with the started provider's options.</summary>
    private InferenceSession Session(OcrNetworkKind kind)
    {
        lock (_lock)
        {
            return _sessions[(int)kind] ??= new InferenceSession(ModelData.Load(kind), _sessionOptions!);
        }
    }

    /// <summary>
    /// Picks the provider once: the first of the candidates whose session options can be made and that loads the first
    /// network this accelerator runs, which proves that the provider is in the native build and that a device answers.
    /// </summary>
    private void Start()
    {
        lock (_lock)
        {
            if (_started)
            {
                if (_active is null && !_fallBackToCpu) throw new InvalidOperationException(_fallbackReason);
                return;
            }

            _started = true;
            if (_networks == OcrNetworks.None)
            {
                _fallbackReason = "No networks were chosen to run on the GPU.";
                return;
            }

            OcrNetworkKind first = (_networks & OcrNetworks.Detection) != 0 ? OcrNetworkKind.Detection
                : (_networks & OcrNetworks.Recognition) != 0 ? OcrNetworkKind.Recognition : OcrNetworkKind.Orientation;
            GpuProvider[] candidates = _provider == GpuProvider.Auto ? [GpuProvider.Cuda, GpuProvider.DirectML] : [_provider];
            var reasons = new List<string>();
            foreach (GpuProvider candidate in candidates)
            {
                SessionOptions? options = null;
                try
                {
                    options = CreateSessionOptions(candidate, _deviceId);
                    _sessions[(int)first] = new InferenceSession(ModelData.Load(first), options);
                    _sessionOptions = options;
                    _active = candidate;
                    return;
                }
                catch (Exception e) when (IsStartFailure(e))
                {
                    options?.Dispose();
                    reasons.Add($"{candidate}: {e.GetType().Name}: {e.Message}");
                }
            }

            _fallbackReason = "No ONNX Runtime execution provider could be started (" + string.Join("; ", reasons) + ").";
            if (!_fallBackToCpu) throw new InvalidOperationException(_fallbackReason);
        }
    }

    private static SessionOptions CreateSessionOptions(GpuProvider provider, int deviceId)
    {
        var options = new SessionOptions();
        try
        {
            switch (provider)
            {
                case GpuProvider.Cuda:
                    using (var cuda = new OrtCUDAProviderOptions())
                    {
                        // The pages and line batches come in many sizes, and the exhaustive search cuDNN does by default
                        // for every new shape costs more than the convolution it speeds up.
                        cuda.UpdateOptions(new Dictionary<string, string>
                        {
                            ["device_id"] = deviceId.ToString(System.Globalization.CultureInfo.InvariantCulture),
                            ["cudnn_conv_algo_search"] = "HEURISTIC",
                        });
                        options.AppendExecutionProvider_CUDA(cuda);
                    }

                    break;
                case GpuProvider.DirectML:
                    // DirectML needs sequential execution and no memory pattern, as ONNX Runtime's documentation says.
                    options.EnableMemoryPattern = false;
                    options.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
                    options.AppendExecutionProvider_DML(deviceId);
                    break;
                case GpuProvider.OnnxRuntimeCpu:
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(provider), provider, "Not a provider that can be started.");
            }

            return options;
        }
        catch
        {
            options.Dispose();
            throw;
        }
    }

    /// <summary>Whether an exception means that ONNX Runtime's native library could not be loaded.</summary>
    private static bool IsLoadFailure(Exception e)
        => e is DllNotFoundException or EntryPointNotFoundException or BadImageFormatException
            || (e is TypeInitializationException { InnerException: { } inner } && IsLoadFailure(inner));

    /// <summary>Whether an exception means that a provider cannot run here: missing from the build, or no device.</summary>
    private static bool IsStartFailure(Exception e) => e is OnnxRuntimeException || IsLoadFailure(e);
}
