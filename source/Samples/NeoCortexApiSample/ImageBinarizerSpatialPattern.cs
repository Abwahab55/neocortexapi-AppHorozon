using System;
using System.IO;
using System.Linq;
using Daenet.Binarizer;
using Daenet.Binarizer.Entities;
using NeoCortexApi;
using NeoCortexApi.Entities;

namespace NeoCortexApiSample
{
    public class ImageBinarizerSpatialPattern
    {
        private string trainingFolder;
        private SpatialPooler spatialPooler;
        private Connections connections;

        public ImageBinarizerSpatialPattern(string trainingFolder)
        {
            this.trainingFolder = trainingFolder;

            connections = new Connections(new HtmConfig
            {
                ColumnDimensions = new int[] { 28, 28 },
                InputDimensions = new int[] { 28, 28 },
                NumInputs = 784,
                PotentialRadius = 16,
                NumActiveColumnsPerInhArea = 40,
                SynPermInactiveDec = 0.005,
                SynPermActiveInc = 0.04,
                SynPermConnected = 0.1,
                PotentialPct = 0.85
            });

            spatialPooler = new SpatialPooler();
            spatialPooler.Init(connections);
        }

        public void Run()
        {
            var trainingImages = Directory.GetFiles(trainingFolder, "*.png");

            string sdrOutputFolder = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "SDR_Values");
            Directory.CreateDirectory(sdrOutputFolder);

            string binarizedImagesFolder = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "BinarizedImages");
            Directory.CreateDirectory(binarizedImagesFolder);

            foreach (var imagePath in trainingImages)
            {
                string imageName = Path.GetFileNameWithoutExtension(imagePath);
                Console.WriteLine($"Processing Image: {imageName}");

                var binarizerParams = new BinarizerParams
                {
                    InputImagePath = imagePath,
                    OutputImagePath = Path.Combine(binarizedImagesFolder, $"{imageName}_binarized.png"),
                    GreyScale = true,
                    ImageWidth = 28,
                    ImageHeight = 28
                };

                var binarizer = new ImageBinarizer(binarizerParams);
                binarizer.Run();

                int[] binarizedPixels = File.ReadAllLines(binarizerParams.OutputImagePath)
                    .SelectMany(line => line.Trim().Select(c => c - '0'))
                    .ToArray();

                int[] activeColumns = new int[connections.HtmConfig.NumColumns];
                spatialPooler.compute(binarizedPixels, activeColumns, true);

                File.WriteAllText(Path.Combine(sdrOutputFolder, $"input_{imageName}.txt"), string.Join(",", activeColumns));
                Console.WriteLine($"SDR Active Bits for {imageName}: {activeColumns.Count(b => b == 1)}");
            }

            Console.WriteLine("Image Processing Completed.");
        }
    }
}
