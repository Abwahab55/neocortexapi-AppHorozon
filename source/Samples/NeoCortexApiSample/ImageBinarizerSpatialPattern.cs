using System;
using System.IO;
using System.Linq;
using System.Drawing;
using Daenet.ImageBinarizerLib;
using Daenet.ImageBinarizerLib.Entities;
using NeoCortexApi;
using NeoCortexApi.Entities;

namespace NeoCortexApiSample
{
    /// <summary>
    /// Loads images, binarizes them with Daenet, encodes using Spatial Pooler, and generates valid PNGs.
    /// </summary>
    public class ImageBinarizerSpatialPattern
    {
        private readonly string trainingFolder;
        private readonly SpatialPooler spatialPooler;
        private readonly Connections connections;
        private const int ImageSize = 64;

        public ImageBinarizerSpatialPattern(string trainingFolder)
        {
            if (string.IsNullOrEmpty(trainingFolder) || !Directory.Exists(trainingFolder))
                throw new ArgumentException("Training folder path is invalid or does not exist.");

            this.trainingFolder = trainingFolder;

            connections = new Connections(new HtmConfig
            {
                ColumnDimensions = new[] { ImageSize, ImageSize },
                InputDimensions = new[] { ImageSize, ImageSize },
                NumInputs = ImageSize * ImageSize,
                PotentialRadius = ImageSize,
                PotentialPct = 0.85,
                NumActiveColumnsPerInhArea = 100,
                SynPermInactiveDec = 0.005,
                SynPermActiveInc = 0.04,
                SynPermConnected = 0.2
            });

            spatialPooler = new SpatialPooler();
            spatialPooler.Init(connections);
        }

        public void Run()
        {
            string[] images = Directory.GetFiles(trainingFolder, "*.png");
            if (images.Length == 0)
            {
                Console.WriteLine("No images found in the training folder.");
                return;
            }

            string sdrOutputFolder = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "SDR_Values");
            Directory.CreateDirectory(sdrOutputFolder);

            Console.WriteLine("Running Image Binarization and Encoding...");

            foreach (var imagePath in images)
            {
                string imageName = Path.GetFileNameWithoutExtension(imagePath);

                // Binarize image and get pixels from txt file directly
                int[] binarizedPixels = BinarizeImage(imagePath, sdrOutputFolder, imageName);

                if (binarizedPixels == null || binarizedPixels.Length != ImageSize * ImageSize)
                {
                    Console.WriteLine($"Error: {imageName} SDR incorrect size.");
                    continue;
                }

                // Generate SDR
                int[] activeColumns = new int[connections.HtmConfig.NumColumns];
                spatialPooler.compute(binarizedPixels, activeColumns, learn: true);
                File.WriteAllText(Path.Combine(sdrOutputFolder, $"{imageName}.txt"), string.Join(",", activeColumns));

                Console.WriteLine($"Processed SDR for {imageName}. Active bits: {activeColumns.Count(bit => bit == 1)}");
            }

            Console.WriteLine("Image Binarization and Encoding completed successfully.");
        }

        private int[] BinarizeImage(string inputImagePath, string outputFolder, string imageName)
        {
            if (!File.Exists(inputImagePath))
            {
                Console.WriteLine($"Error: Image not found at path {inputImagePath}");
                return null;
            }

            string binarizedTxtPath = Path.Combine(outputFolder, $"{imageName}_binarized.txt");
            string binarizedPngPath = Path.Combine(outputFolder, $"{imageName}_binarized.png");

            try
            {
                var binParams = new BinarizerParams
                {
                    InputImagePath = inputImagePath,
                    OutputImagePath = binarizedTxtPath,
                    GreyScale = true,
                    ImageWidth = ImageSize,
                    ImageHeight = ImageSize,
                    GreyThreshold = 128
                };

                // Run binarizer (outputs text file)
                new ImageBinarizer(binParams).Run();

                if (!File.Exists(binarizedTxtPath))
                {
                    Console.WriteLine($"Error: Binarized file was not created at {binarizedTxtPath}");
                    return null;
                }

                // Read the binary data directly from text file
                var lines = File.ReadAllLines(binarizedTxtPath);
                if (lines.Length != ImageSize)
                {
                    Console.WriteLine("Error: Binarized txt file incorrect dimensions.");
                    return null;
                }

                int[] pixels = lines.SelectMany(line => line.Select(ch => ch == '1' ? 1 : 0)).ToArray();

                // Create and save a valid PNG for reference
                SaveBitmapFromPixels(pixels, binarizedPngPath);

                return pixels;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error binarizing image {imageName}: {ex.Message}");
                return null;
            }
        }

        private void SaveBitmapFromPixels(int[] pixels, string filePath)
        {
            using Bitmap bmp = new Bitmap(ImageSize, ImageSize);
            for (int y = 0; y < ImageSize; y++)
            {
                for (int x = 0; x < ImageSize; x++)
                {
                    int pixelValue = pixels[y * ImageSize + x];
                    Color color = pixelValue == 1 ? Color.Black : Color.White;
                    bmp.SetPixel(x, y, color);
                }
            }
            bmp.Save(filePath, System.Drawing.Imaging.ImageFormat.Png);
        }
    }
}
