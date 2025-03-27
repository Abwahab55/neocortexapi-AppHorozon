using System;
using System.Collections.Generic;
using System.Linq;
using NeoCortexApi;
using NeoCortexApi.Classifiers;
using NeoCortexApi.Entities;
using NeoCortexApi.Utility;

namespace NeoCortexApiSample
{
    public class HtmImageClassifier : IClassifier<int[], string>
    {
        private readonly Dictionary<string, int[]> trainingData = new();
        private readonly TemporalMemory tm;

        public HtmImageClassifier(int width = 64, int height = 64)
        {
            var config = new HtmConfig
            {
                ColumnDimensions = new[] { width, height },
                InputDimensions = new[] { width, height },
                NumInputs = width * height,
                PotentialPct = 0.6,
                SynPermConnected = 0.2
            };

            var connections = new Connections(config);
            tm = new TemporalMemory();
            tm.Init(connections);
        }

        public void Learn(int[] input, Cell[] _)
        {
            tm.Compute(input, learn: true);
            string hash = string.Join("", input.Select(i => i.ToString()));
            if (!trainingData.ContainsKey(hash))
                trainingData[hash] = input;
        }

        public int[] GetPredictedInputValue(Cell[] cells) =>
            GetPredictedInputValues(cells.Select(c => c.Index).ToArray(), 1)
                .FirstOrDefault()?.PredictedInput;

        public List<ClassifierResult<int[]>> GetPredictedInputValues(int[] sdr, short n)
        {
            tm.Compute(sdr, learn: false);

            var results = trainingData.Values.Select(stored => new ClassifierResult<int[]>
            {
                PredictedInput = stored,
                Similarity = stored.Zip(sdr, (a, b) => a == b ? 1 : 0).Sum()
            })
            .OrderByDescending(x => x.Similarity)
            .Take(n)
            .ToList();

            foreach (var res in results)
            {
                Console.WriteLine("\n[HTM Prediction Similarity Metrics]");
                PrintSimilarityMetrics(sdr, res.PredictedInput);
            }

            return results;
        }

        private void PrintSimilarityMetrics(int[] original, int[] prediction)
        {
            double jaccard = MathHelpers.JaccardSimilarityofBinaryArrays(original, prediction);
            double cosine = ComputeCosineSimilarity(original, prediction);
            double hamming = ComputeHammingSimilarity(original, prediction);
            double hybrid = (jaccard + cosine + hamming) / 3.0;

            Console.WriteLine($"  Cosine:  {cosine:F4}");
            Console.WriteLine($"  Jaccard: {jaccard:F4}");
            Console.WriteLine($"  Hamming: {hamming:F4}");
            Console.WriteLine($"  Hybrid:  {hybrid:F4}");
        }

        private double ComputeCosineSimilarity(int[] a, int[] b)
        {
            double dot = a.Zip(b, (x, y) => x * y).Sum();
            double magA = Math.Sqrt(a.Sum(x => x * x));
            double magB = Math.Sqrt(b.Sum(y => y * y));
            return (magA == 0 || magB == 0) ? 0.0 : dot / (magA * magB);
        }

        private double ComputeHammingSimilarity(int[] a, int[] b)
        {
            return a.Zip(b, (x, y) => x == y ? 1 : 0).Sum() / (double)a.Length;
        }
    }
}
