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
                PotentialRadius = 28,
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
                    OutputImagePath = Path.Combine(binarizedImagesFolder, $"{imageName}_binarized.txt"),
                    GreyScale = true,
                    ImageWidth = 28,
                    ImageHeight = 28  // resize to 28x28 to match HTM input size
                };

                var binarizer = new ImageBinarizer(binParams);
                binarizer.Run();

                // Read the binarized image (0/1 text format) into an array
                int[] binarizedPixels = File.ReadAllLines(binParams.OutputImagePath)
                                            .SelectMany(line => line.Trim().Select(c => c - '0'))
                                            .ToArray();

                // Apply Spatial Pooler to get SDR of active columns
                int[] activeColumns = new int[connections.HtmConfig.NumColumns];
                spatialPooler.compute(binarizedPixels, activeColumns, true);

                // Save the raw SDR (active columns as 0/1) to a file
                File.WriteAllText(Path.Combine(sdrOutputFolder, $"{imageName}.txt"),
                                  string.Join(",", activeColumns));
                Console.WriteLine($"Processed SDR for {imageName}. Active bits: {activeColumns.Count(bit => bit == 1)}");
            }
        }

        // (NormalizeSdr method is no longer used; we preserve it here for reference)
        private static int[] NormalizeSdr(int[] sdr, double density = 0.25)
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
