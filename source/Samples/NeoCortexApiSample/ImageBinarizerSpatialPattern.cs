using System;
using System.IO;
using System.Linq;
using System.Drawing;
using NeoCortexApi;
using NeoCortexApi.Entities;

namespace NeoCortexApiSample
{
    public class ImageBinarizerSpatialPattern
    {
        private string trainingFolder;
        private SpatialPooler spatialPooler;
        private Connections connections;
        private const int ImageSize = 28;

        public ImageBinarizerSpatialPattern(string trainingFolder)
        {
            if (string.IsNullOrEmpty(trainingFolder) || !Directory.Exists(trainingFolder))
                throw new ArgumentException("Invalid training folder path.");

            this.trainingFolder = trainingFolder;

            connections = new Connections(new HtmConfig
            {
                ColumnDimensions = new int[] { ImageSize, ImageSize },
                InputDimensions = new int[] { ImageSize, ImageSize },
                NumInputs = ImageSize * ImageSize,
                PotentialRadius = ImageSize,
                NumActiveColumnsPerInhArea = 30,
                SynPermInactiveDec = 0.005,
                SynPermActiveInc = 0.04,
                SynPermConnected = 0.2,
                PotentialPct = 0.85
            });

            spatialPooler = new SpatialPooler();
            spatialPooler.Init(connections);
        }

        public void Run()
        {
            var images = Directory.GetFiles(trainingFolder, "*.png");
            if (images.Length == 0)
            {
                Console.WriteLine("No images found in the training folder.");
                return;
            }

            string sdrOutputFolder = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "SDR_Values");
            Directory.CreateDirectory(sdrOutputFolder);

            foreach (var imagePath in images)
            {
                string imageName = Path.GetFileNameWithoutExtension(imagePath);
                int[] binarizedPixels = BinarizeImage(imagePath, sdrOutputFolder, imageName);

                if (binarizedPixels == null || binarizedPixels.Length != ImageSize * ImageSize)
                {
                    Console.WriteLine($"Error: {imageName} SDR incorrect size. Expected {ImageSize * ImageSize}.");
                    continue;
                }

                int[] activeColumns = new int[connections.HtmConfig.NumColumns];
                spatialPooler.compute(binarizedPixels, activeColumns, true);

                File.WriteAllText(Path.Combine(sdrOutputFolder, $"{imageName}.txt"), string.Join(",", activeColumns));
                Console.WriteLine($"Processed SDR for {imageName}. Active bits: {activeColumns.Count(bit => bit == 1)}");
            }
        }

        private int[] BinarizeImage(string inputImagePath, string outputFolder, string imageName)
        {
            if (!File.Exists(inputImagePath))
            {
                Console.WriteLine($"Error: Image not found - {inputImagePath}");
                return null;
            }

            string outputImagePath = Path.Combine(outputFolder, $"{imageName}_binarized.png");

            try
            {
                using (Bitmap originalImage = new Bitmap(inputImagePath))
                {
                    int newWidth = ImageSize;
                    int newHeight = ImageSize;

                    using (Bitmap resizedImage = new Bitmap(originalImage, new Size(newWidth, newHeight)))
                    using (Bitmap binarizedImage = new Bitmap(newWidth, newHeight))
                    {
                        for (int x = 0; x < newWidth; x++)
                        {
                            for (int y = 0; y < newHeight; y++)
                            {
                                Color pixelColor = resizedImage.GetPixel(x, y);
                                int grayscale = (pixelColor.R + pixelColor.G + pixelColor.B) / 3;
                                Color binaryColor = (grayscale < 128) ? Color.Black : Color.White;
                                binarizedImage.SetPixel(x, y, binaryColor);
                            }
                        }

                        binarizedImage.Save(outputImagePath, System.Drawing.Imaging.ImageFormat.Png);
                    }
                }

                return ReadBinarizedImage(outputImagePath);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error during binarization: {ex.Message}");
                return null;
            }
        }

        private int[] ReadBinarizedImage(string imagePath)
        {
            try
            {
                using (Bitmap bmp = new Bitmap(imagePath))
                {
                    int[] binaryPixels = new int[ImageSize * ImageSize];
                    for (int y = 0; y < bmp.Height; y++)
                    {
                        for (int x = 0; x < bmp.Width; x++)
                        {
                            Color pixel = bmp.GetPixel(x, y);
                            int bit = pixel.R < 128 ? 1 : 0;
                            binaryPixels[y * ImageSize + x] = bit;
                        }
                    }
                    return binaryPixels;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error reading binarized image: {ex.Message}");
                return null;
            }
        }
    }
}
