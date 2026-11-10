namespace PdfPecker.Ocr.Gpu.Tests;

/// <summary>Synthetic pages drawn by PdfPecker from HTML and rendered at 300 dpi, as a clean scan would look.</summary>
internal static class TestPages
{
    public const string InvoiceText =
        "Invoice 2026-0412\nExample Customer Ltd.\nStorgata 12, 0155 Oslo\nDate: 8 November 2026\nConsulting, October 11,400.00\nTotal NOK 17,589.45";

    private static readonly Lazy<(byte[] Rgba, int Width, int Height)> s_invoice = new(() => Render(InvoiceText));

    public static (byte[] Rgba, int Width, int Height) Invoice => s_invoice.Value;

    public static (byte[] Rgba, int Width, int Height) Render(string text)
    {
        string body = string.Concat(text.Split('\n').Select(l => "<p>" + System.Net.WebUtility.HtmlEncode(l) + "</p>"));
        string html = "<!doctype html><html><head><meta charset=\"utf-8\"><style>@page{size:150mm 80mm;margin:8mm}body{margin:0}"
            + "p{font:12pt/1.5 sans-serif;margin:0}</style></head><body>" + body + "</body></html>";
        byte[] pdf = HtmlToPdf.Convert(html);
        var image = PdfRenderer.Open(pdf).RenderPage(0, new ImageOptions { Dpi = 300 });
        return (image.Rgba, image.Width, image.Height);
    }
}
