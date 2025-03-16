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
                PotentialRadius = 28,  // Entire input coverage
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

            string sdrOutputFolder = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "SDR_Values");
            Directory.CreateDirectory(sdrOutputFolder);

            string binarizedImagesFolder = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "BinarizedImages");
            Directory.CreateDirectory(binarizedImagesFolder);

            foreach (var imagePath in images)
            {
                string imageName = Path.GetFileNameWithoutExtension(imagePath);

                var binParams = new BinarizerParams
                {
                    InputImagePath = imagePath,
                    OutputImagePath = Path.Combine(binarizedImagesFolder, $"{imageName}_binarized.png"),
                    GreyScale = true,
                    ImageWidth = 28,
                    ImageHeight = 28
                };

                var binarizer = new ImageBinarizer(binParams);
                binarizer.Run();

                int[] binarizedPixels = File.ReadAllLines(binParams.OutputImagePath)
                    .SelectMany(line => line.Trim().Select(c => c - '0'))
                    .ToArray();

                // Apply Spatial Pooler
                int[] activeColumns = new int[connections.HtmConfig.NumColumns];
                spatialPooler.compute(binarizedPixels, activeColumns, true);

                // Normalize SDR to desired sparsity
                int[] normalizedSDR = NormalizeSdr(activeColumns, density: 0.2);

                File.WriteAllText(Path.Combine(sdrOutputFolder, $"{imageName}.txt"), string.Join(",", normalizedSDR));
                Console.WriteLine($"Processed SDR for {imageName}. Active bits: {normalizedSDR.Count(bit => bit == 1)}");
            }
        }

        private static int[] NormalizeSdr(int[] sdr, double density = 0.2)
        {
            int activeBitsCount = (int)(sdr.Length * density);

            var sortedIndices = sdr
                .Select((value, index) => new { value, index })
                .OrderByDescending(x => x.value)
                .Take(activeBitsCount)
                .Select(x => x.index)
                .ToArray();

            int[] normalizedSDR = new int[sdr.Length];
            foreach (int idx in sortedIndices)
            {
                normalizedSDR[idx] = 1;
            }

            return normalizedSDR;
        }
    }
}
