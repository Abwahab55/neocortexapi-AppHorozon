using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeoCortexApiSample;
using NeoCortexApi.Classifiers;
using System.IO;

namespace NeoCortexApiSample1.Tests
{
    [TestClass]
    public class KnnImageReconstructorTests
    {
        private string sdrFolder;
        private string outputFolder;
        private string reconstructedSdrFolder;

        // This method runs before each test to set up temporary test folders and input SDR data.
        [TestInitialize]
        public void Setup()
        {
            // Define temporary folders for input and output
            sdrFolder = Path.Combine(Path.GetTempPath(), "SDR");
            outputFolder = Path.Combine(Path.GetTempPath(), "Output");
            reconstructedSdrFolder = Path.Combine(Path.GetTempPath(), "Reconstructed");

            // Create input SDR folder and add a dummy SDR file (64x64 flat image with all zeros)
            Directory.CreateDirectory(sdrFolder);
            File.WriteAllText(Path.Combine(sdrFolder, "test.txt"), string.Join(",", new int[64 * 64]));
        }

        // This test verifies that the k-NN image reconstructor generates the expected output files.
        [TestMethod]
        public void RunReconstruction_CreatesFiles()
        {
            // Create and train a simple k-NN classifier with a dummy input
            var classifier = new KnnImageClassifier();
            classifier.Learn(new int[64 * 64], null);

            // Create the reconstructor and run the reconstruction process
            var reconstructor = new KnnImageReconstructor(64, 64, 1);
            reconstructor.RunReconstruction(sdrFolder, outputFolder, reconstructedSdrFolder, classifier);

            // Assert that the output folder contains at least one file (e.g., reconstructed image)
            Assert.IsTrue(Directory.GetFiles(outputFolder).Length > 0, "No files were created in the output folder.");

            // Assert that the reconstructed SDR folder also contains output (e.g., .txt SDR)
            Assert.IsTrue(Directory.GetFiles(reconstructedSdrFolder).Length > 0, "No files were created in the reconstructed SDR folder.");
        }

        // This method cleans up all temporary folders and files after each test run.
        [TestCleanup]
        public void Cleanup()
        {
            Directory.Delete(sdrFolder, true);
            Directory.Delete(outputFolder, true);
            Directory.Delete(reconstructedSdrFolder, true);
        }
    }
}
