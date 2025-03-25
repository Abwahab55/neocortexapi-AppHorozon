using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeoCortexApiSample;
using NeoCortexApi.Entities;
using System.Linq;

namespace NeoCortexApiSample.Tests
{
    [TestClass]
    public class HtmImageClassifierTests
    {
        [TestMethod]
        public void Learn_ShouldNotThrow()
        {
            var classifier = new HtmImageClassifier();
            var input = Enumerable.Repeat(1, 64 * 64).ToArray();
            classifier.Learn(input, new Cell[0]);
        }

        [TestMethod]
        public void GetPredictedInputValues_ReturnsSimilarity()
        {
            var classifier = new HtmImageClassifier();
            var input = Enumerable.Repeat(1, 64 * 64).ToArray();
            classifier.Learn(input, new Cell[0]);

            var predictions = classifier.GetPredictedInputValues(input, 1);
            Assert.IsTrue(predictions.Count > 0);
        }
    }
}
