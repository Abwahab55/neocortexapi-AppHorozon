using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeoCortexApi.Entities;
using NeoCortexApiSample;
using System.Linq;

[TestClass]
public class HtmImageClassifierTests
{
    // This test verifies that calling Learn() on the classifier doesn't throw any exceptions.
    // It's a basic sanity check to ensure that the classifier can accept input without crashing.
    [TestMethod]
    public void Learn_ShouldNotThrow()
    {
        // Create a new instance of the classifier
        var classifier = new HtmImageClassifier();

        // Simulate an input SDR (e.g., a flattened 64x64 binary image)
        var input = Enumerable.Repeat(1, 64 * 64).ToArray();

        // Create an array of empty Cell objects with the same length
        var cells = new Cell[input.Length];

        // Attempt to teach the classifier using the input and cells
        classifier.Learn(input, cells);
    }

    // This test checks whether the classifier can make accurate predictions after learning.
    // It verifies that the predicted output has a high similarity with the original input.
    [TestMethod]
    public void GetPredictedInputValues_ReturnsHighSimilarity()
    {
        // Initialize the classifier
        var classifier = new HtmImageClassifier();

        // Create a dummy input SDR
        var input = Enumerable.Repeat(1, 64 * 64).ToArray();

        // Create a matching array of Cell objects
        var cells = new Cell[input.Length];

        // Train the classifier with the input SDR
        classifier.Learn(input, cells);

        // Ask the classifier to predict based on the input SDR and return 1 top prediction
        var predictions = classifier.GetPredictedInputValues(input, 1);

        // Make sure we actually got a prediction
        Assert.IsNotNull(predictions);
        Assert.IsTrue(predictions.Count > 0);

        // Extract the predicted SDR from the top result
        var predicted = predictions[0].PredictedInput;

        // Check that the predicted SDR has the same length as the original input
        Assert.AreEqual(input.Length, predicted.Length);

        // Calculate how many bits match between the input and predicted SDR
        int overlap = input.Zip(predicted, (a, b) => a == b ? 1 : 0).Sum();

        // Compute the similarity ratio (e.g., 0.95 = 95% match)
        double similarityRatio = (double)overlap / input.Length;

        // The prediction should be at least 70% similar to the input
        Assert.IsTrue(similarityRatio > 0.7, $"Similarity too low: {similarityRatio:F2}");
    }
}
