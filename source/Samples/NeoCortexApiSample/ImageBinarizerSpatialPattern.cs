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
    public class ImageBinarizerSpatialPattern
    {
        private readonly string trainingFolder;
        private readonly SpatialPooler spatialPooler;
        private readonly Connections connections;
        private const int ImageSize = 64;

        public ImageBinarizerSpatialPattern(string trainingFolder)
        {
            if (!Directory.Exists(trainingFolder))
                throw new ArgumentException("Invalid training folder path.");

            this.trainingFolder = trainingFolder;

            connections = new Connections(new HtmConfig
            {
                ColumnDimensions = new[] { ImageSize, ImageSize },
                InputDimensions = new[] { ImageSize, ImageSize },
                NumInputs = ImageSize * ImageSize,
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
            var images = Directory.GetFiles(trainingFolder, "*.png");
            var sdrOutputFolder = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "SDR_Values");
            Directory.CreateDirectory(sdrOutputFolder);

            foreach (var imagePath in images)
            {
                var imageName = Path.GetFileNameWithoutExtension(imagePath);
                var binarizedPixels = BinarizeImage(imagePath, sdrOutputFolder, imageName);
                if (binarizedPixels == null) continue;

                var activeColumns = new int[connections.HtmConfig.NumColumns];
                spatialPooler.compute(binarizedPixels, activeColumns, true);
                File.WriteAllText(Path.Combine(sdrOutputFolder, $"{imageName}.txt"),
    string.Join(",", activeColumns.Select(bit => bit.ToString())));
            }
        }

        private int[] BinarizeImage(string path, string folder, string name)
        {
            var binParams = new BinarizerParams
            {
                InputImagePath = path,
                OutputImagePath = Path.Combine(folder, $"{name}_binarized.txt"),
                GreyScale = true,
                ImageWidth = ImageSize,
                ImageHeight = ImageSize,
                GreyThreshold = 128
            };

            new ImageBinarizer(binParams).Run();
            var lines = File.ReadAllLines(binParams.OutputImagePath);
            return lines.SelectMany(line => line.Select(ch => ch == '1' ? 1 : 0)).ToArray();
        }
    }
}
