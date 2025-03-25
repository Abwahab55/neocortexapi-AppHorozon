using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeoCortexApiSample;
using NeoCortexApi.Classifiers;
using System.IO;

namespace NeoCortexApiSample.Tests
{
    [TestClass]
    public class KnnImageReconstructorTests
    {
        private string sdrFolder;
        private string outputFolder;
        private string reconstructedSdrFolder;

        [TestInitialize]
        public void Setup()
        {
            sdrFolder = Path.Combine(Path.GetTempPath(), "SDR");
            outputFolder = Path.Combine(Path.GetTempPath(), "Output");
            reconstructedSdrFolder = Path.Combine(Path.GetTempPath(), "Reconstructed");

            Directory.CreateDirectory(sdrFolder);
            File.WriteAllText(Path.Combine(sdrFolder, "test.txt"), string.Join(",", new int[64 * 64]));
        }

        [TestMethod]
        public void RunReconstruction_CreatesFiles()
        {
            var classifier = new KnnImageClassifier();
            classifier.Learn(new int[64 * 64], null);

            var reconstructor = new KnnImageReconstructor(64, 64, 1);
            reconstructor.RunReconstruction(sdrFolder, outputFolder, reconstructedSdrFolder, classifier);

            Assert.IsTrue(Directory.GetFiles(outputFolder).Length > 0);
            Assert.IsTrue(Directory.GetFiles(reconstructedSdrFolder).Length > 0);
        }

        [TestCleanup]
        public void Cleanup()
        {
            Directory.Delete(sdrFolder, true);
            Directory.Delete(outputFolder, true);
            Directory.Delete(reconstructedSdrFolder, true);
        }
    }
}
