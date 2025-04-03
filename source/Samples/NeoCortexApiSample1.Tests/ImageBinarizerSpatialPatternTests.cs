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

        [TestInitialize]
        public void Setup()
        {
            _testImagesDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "TestImages");
            Directory.CreateDirectory(_testImagesDir);

            // Add a dummy binarized test image for safety (you can replace this with an actual small .png file)
            string dummyImage = Path.Combine(_testImagesDir, "sample.png");
            if (!File.Exists(dummyImage))
            {
                using (var bmp = new System.Drawing.Bitmap(64, 64))
                {
                    using (var g = System.Drawing.Graphics.FromImage(bmp))
                    {
                        g.Clear(System.Drawing.Color.White);
                        g.FillRectangle(System.Drawing.Brushes.Black, 0, 0, 32, 32);
                    }
                    bmp.Save(dummyImage, System.Drawing.Imaging.ImageFormat.Png);
                }
            }

            _outputSdrDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "SDR_Values");
            if (Directory.Exists(_outputSdrDir))
                Directory.Delete(_outputSdrDir, true);
        }

        [TestMethod]
        public void Run_ShouldGenerateSdrFiles()
        {
            // Arrange
            var binarizer = new ImageBinarizerSpatialPattern(_testImagesDir);

            // Act
            binarizer.Run();

            // Assert
            Assert.IsTrue(Directory.Exists(_outputSdrDir), "SDR output folder was not created.");
            var sdrFiles = Directory.GetFiles(_outputSdrDir, "*.txt");
            Assert.IsTrue(sdrFiles.Length > 0, "No SDR files were generated.");
            var content = File.ReadAllText(sdrFiles.First());
            Assert.IsFalse(string.IsNullOrWhiteSpace(content), "Generated SDR file is empty.");
        }

        [TestCleanup]
        public void Cleanup()
        {
            if (Directory.Exists(_testImagesDir))
                Directory.Delete(_testImagesDir, true);

            if (Directory.Exists(_outputSdrDir))
                Directory.Delete(_outputSdrDir, true);
        }
    }
}
