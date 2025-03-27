using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeoCortexApiSample;
using NeoCortexApi.Entities;
using System.Linq;

[TestClass]
public class KnnImageClassifierTests
{
    [TestMethod]
    public void Learn_And_Predict_ShouldWork()
    {
        var classifier = new KnnImageClassifier(k: 3);
        var sdr = Enumerable.Repeat(1, 64 * 64).ToArray();
        classifier.Learn(sdr, new Cell[sdr.Length]);

        var prediction = classifier.GetPredictedInputValues(sdr, 1);
        Assert.IsNotNull(prediction);
        Assert.AreEqual(sdr.Length, prediction.First().PredictedInput.Length);
    }
}
