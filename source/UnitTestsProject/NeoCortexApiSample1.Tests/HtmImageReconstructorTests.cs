using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeoCortexApiSample;
using NeoCortexApi.Classifiers;
using System.IO;

namespace NeoCortexApiSample1.Tests
{
    [TestClass]
    public class HtmImageReconstructorTests
    {
        private string sdrFolder;
        private string outputFolder;
        private string reconstructedSdrFolder;

        // This method runs before each test.
        // It sets up temporary folders and a dummy SDR file for reconstruction.
        [TestInitialize]
        public void Setup()
        {
            // Create temporary folders for input and output
            sdrFolder = Path.Combine(Path.GetTempPath(), "SDR_HTM");
            outputFolder = Path.Combine(Path.GetTempPath(), "Output_HTM");
            reconstructedSdrFolder = Path.Combine(Path.GetTempPath(), "Reconstructed_HTM");

            Directory.CreateDirectory(sdrFolder);

            // Create a dummy SDR input file with zeros (flat image)
            File.WriteAllText(Path.Combine(sdrFolder, "sample.txt"), string.Join(",", new int[64 * 64]));
        }

        // This test verifies that the HTM reconstruction process creates the expected output files.
        [TestMethod]
        public void RunReconstruction_GeneratesFiles()
        {
            // Create a dummy classifier and learn a blank image
            var classifier = new HtmImageClassifier();
            classifier.Learn(new int[64 * 64], null);

            // Initialize the reconstructor and run the reconstruction
            var reconstructor = new HtmImageReconstructor();
            reconstructor.RunReconstruction(
                sdrFolder,                  // Folder containing input SDR
                outputFolder,              // Where to save the reconstructed image
                reconstructedSdrFolder,    // Where to save the reconstructed SDR
                classifier,                // Classifier to use for prediction
                64, 64                     // Image dimensions
            );

            // Check that the expected output image file exists
            Assert.IsTrue(File.Exists(Path.Combine(outputFolder, "sample_HTM_Reconstructed.png")));

            // Check that the expected reconstructed SDR file exists
            Assert.IsTrue(File.Exists(Path.Combine(reconstructedSdrFolder, "sample_HTM_Reconstructed.txt")));
        }

        // This method runs after each test.
        // It deletes all temporary files and folders created during the test.
        [TestCleanup]
        public void Cleanup()
        {
            Directory.Delete(sdrFolder, true);
            Directory.Delete(outputFolder, true);
            Directory.Delete(reconstructedSdrFolder, true);
        }
    }
}
