using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeoCortexApi.Entities;
using NeoCortexApiSample;
using System.Linq;

[TestClass]
public class HtmImageClassifierTests
{
    [TestMethod]
    public void Learn_ShouldNotThrow()
    {
        var classifier = new HtmImageClassifier();
        var input = Enumerable.Repeat(1, 64 * 64).ToArray();
        var cells = new Cell[input.Length];

        classifier.Learn(input, cells);
    }

    [TestMethod]
    public void GetPredictedInputValues_ReturnsHighSimilarity()
    {
        var classifier = new HtmImageClassifier();
        var input = Enumerable.Repeat(1, 64 * 64).ToArray();
        var cells = new Cell[input.Length];

        classifier.Learn(input, cells);

        var predictions = classifier.GetPredictedInputValues(input, 1);

        Assert.IsNotNull(predictions);
        Assert.IsTrue(predictions.Count > 0);

        var predicted = predictions[0].PredictedInput;
        Assert.AreEqual(input.Length, predicted.Length);

        int overlap = input.Zip(predicted, (a, b) => a == b ? 1 : 0).Sum();
        double similarityRatio = (double)overlap / input.Length;

        Assert.IsTrue(similarityRatio > 0.7, $"Similarity too low: {similarityRatio:F2}");
    }
}
