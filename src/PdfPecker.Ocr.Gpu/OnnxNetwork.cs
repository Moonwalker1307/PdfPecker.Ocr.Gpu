using Microsoft.ML.OnnxRuntime;

namespace PdfPecker.Ocr.Gpu;

/// <summary>
/// One network of a worker on an ONNX Runtime session that the accelerator shares between workers, since a session's
/// <c>Run</c> is thread-safe and loading one onto a GPU is slow. The input is copied into a buffer the network keeps, so
/// ONNX Runtime can read managed memory that stays put for the run, and the output is copied out of ONNX Runtime's
/// tensor into another, which stays valid until the next run as the seam asks.
/// </summary>
internal sealed class OnnxNetwork : IOcrNetwork
{
    private readonly InferenceSession _session;
    private readonly string[] _inputNames;
    private readonly string[] _outputNames;
    private readonly RunOptions _runOptions = new();
    private float[] _input = [];
    private float[] _output = [];

    public OnnxNetwork(InferenceSession session)
    {
        _session = session;
        _inputNames = [session.InputNames[0]];
        _outputNames = [session.OutputNames[0]];
    }

    public ReadOnlySpan<float> Run(ReadOnlySpan<float> input, ReadOnlySpan<int> shape, out int[] outputShape)
    {
        if (_input.Length < input.Length)
        {
            _input = new float[input.Length];
        }

        input.CopyTo(_input);
        var dims = new long[shape.Length];
        for (int i = 0; i < dims.Length; i++)
        {
            dims[i] = shape[i];
        }

        using OrtValue value = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, _input.AsMemory(0, input.Length), dims);
        using IDisposableReadOnlyCollection<OrtValue> results = _session.Run(_runOptions, _inputNames, [value], _outputNames);
        OrtValue result = results[0];
        long[] resultShape = result.GetTensorTypeAndShape().Shape;
        ReadOnlySpan<float> data = result.GetTensorDataAsSpan<float>();
        if (_output.Length < data.Length)
        {
            _output = new float[data.Length];
        }

        data.CopyTo(_output);
        outputShape = new int[resultShape.Length];
        for (int i = 0; i < outputShape.Length; i++)
        {
            outputShape[i] = checked((int)resultShape[i]);
        }

        return _output.AsSpan(0, data.Length);
    }

    /// <summary>Lets the run options go; the session belongs to the accelerator and stays.</summary>
    public void Dispose() => _runOptions.Dispose();
}
