using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.IO;
using System.Linq;
using NeoCortexApiSample;

namespace NeoCortexApiSample1.Tests
{
    [TestClass]
    public class ImageBinarizerSpatialPatternTests
    {
        private string _testImagesDir;
        private string _outputSdrDir;

        // This method runs before each test to set up necessary test files and directories.
        [TestInitialize]
        public void Setup()
        {
            // Create a temporary folder for test images
            _testImagesDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "TestImages");
            Directory.CreateDirectory(_testImagesDir);

            // Create a dummy 64x64 test image with a black square in the top-left corner
            // This simulates a simple input image to pass through the binarizer
            string dummyImage = Path.Combine(_testImagesDir, "sample.png");
            if (!File.Exists(dummyImage))
            {
                using (var bmp = new System.Drawing.Bitmap(64, 64))
                {
                    using (var g = System.Drawing.Graphics.FromImage(bmp))
                    {
                        g.Clear(System.Drawing.Color.White);
                        g.FillRectangle(System.Drawing.Brushes.Black, 0, 0, 32, 32); // Half black
                    }
                    bmp.Save(dummyImage, System.Drawing.Imaging.ImageFormat.Png);
                }
            }

            // Define the folder where SDR files will be saved
            _outputSdrDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "SDR_Values");

            // Ensure the folder is clean before each test
            if (Directory.Exists(_outputSdrDir))
                Directory.Delete(_outputSdrDir, true);
        }

        // This test checks that the binarizer successfully creates SDR files from images.
        [TestMethod]
        public void Run_ShouldGenerateSdrFiles()
        {
            // Arrange: create an instance of the binarizer and point it to the test image folder
            var binarizer = new ImageBinarizerSpatialPattern(_testImagesDir);

            // Act: run the binarizer to process images and generate SDRs
            binarizer.Run();

            // Assert: verify that output SDR folder was created
            Assert.IsTrue(Directory.Exists(_outputSdrDir), "SDR output folder was not created.");

            // Assert: check that at least one SDR file was generated
            var sdrFiles = Directory.GetFiles(_outputSdrDir, "*.txt");
            Assert.IsTrue(sdrFiles.Length > 0, "No SDR files were generated.");

            // Assert: check that the generated SDR file is not empty
            var content = File.ReadAllText(sdrFiles.First());
            Assert.IsFalse(string.IsNullOrWhiteSpace(content), "Generated SDR file is empty.");
        }

        // This method runs after each test and cleans up all test data to avoid side effects.
        [TestCleanup]
        public void Cleanup()
        {
            // Delete test image folder
            if (Directory.Exists(_testImagesDir))
                Directory.Delete(_testImagesDir, true);

            // Delete generated SDR output folder
            if (Directory.Exists(_outputSdrDir))
                Directory.Delete(_outputSdrDir, true);
        }
    }
}
