using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeoCortexApiSample;
using NeoCortexApi.Classifiers;
using System.IO;

namespace NeoCortexApiSample.Tests
{
    [TestClass]
    public class HtmImageReconstructorTests
    {
        private string sdrFolder;
        private string outputFolder;
        private string reconstructedSdrFolder;

        [TestInitialize]
        public void Setup()
        {
            sdrFolder = Path.Combine(Path.GetTempPath(), "SDR_HTM");
            outputFolder = Path.Combine(Path.GetTempPath(), "Output_HTM");
            reconstructedSdrFolder = Path.Combine(Path.GetTempPath(), "Reconstructed_HTM");

            Directory.CreateDirectory(sdrFolder);
            File.WriteAllText(Path.Combine(sdrFolder, "sample.txt"), string.Join(",", new int[64 * 64]));
        }

        [TestMethod]
        public void RunReconstruction_GeneratesFiles()
        {
            var classifier = new HtmImageClassifier();
            classifier.Learn(new int[64 * 64], null);

            var reconstructor = new HtmImageReconstructor();
            reconstructor.RunReconstruction(sdrFolder, outputFolder, reconstructedSdrFolder, classifier, 64, 64);

            Assert.IsTrue(File.Exists(Path.Combine(outputFolder, "sample_HTM_Reconstructed.png")));
            Assert.IsTrue(File.Exists(Path.Combine(reconstructedSdrFolder, "sample_HTM_Reconstructed.txt")));
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
