using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeoCortexApiSample;
using NeoCortexApi.Entities;
using System.Linq;

namespace NeoCortexApiSample1.Tests
{
    [TestClass]
    public class KnnImageClassifierTests
    {
        // This test checks the basic functionality of the k-NN classifier:
        // Can it learn a sample SDR and make a prediction afterward?
        [TestMethod]
        public void Learn_And_Predict_ShouldWork()
        {
            // Create a new instance of the k-NN image classifier
            var classifier = new KnnImageClassifier();

            // Create a synthetic SDR (flattened 64x64 image of all 1s)
            var sdr = Enumerable.Repeat(1, 64 * 64).ToArray();

            // Teach the classifier this SDR with dummy cells (unused here)
            classifier.Learn(sdr, new Cell[sdr.Length]);

            // Ask the classifier to predict based on the same input
            var prediction = classifier.GetPredictedInputValues(sdr, 1);

            // Make sure a prediction was returned
            Assert.IsNotNull(prediction);

            // Ensure the predicted SDR has the same length as the input
            Assert.AreEqual(sdr.Length, prediction.First().PredictedInput.Length);
        }
    }
}
