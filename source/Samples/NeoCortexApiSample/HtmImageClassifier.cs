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
        private Dictionary<string, int[]> trainingData = new Dictionary<string, int[]>();
        private TemporalMemory tm;
        private Connections connections;

        public HtmImageClassifier()
        {
            connections = new Connections(new HtmConfig
            {
                ColumnDimensions = new int[] { 28, 28 },
                InputDimensions = new int[] { 28, 28 },
                NumInputs = 784,
                PotentialPct = 0.6,
                SynPermInactiveDec = 0.005,
                SynPermActiveInc = 0.05,   // 🔥 Increased Learning Speed
                SynPermConnected = 0.2,
                NumActiveColumnsPerInhArea = 50 // 🔥 Increased Active Columns
            });

            tm = new TemporalMemory();
            tm.Init(connections);
        }

        public void Learn(int[] input, Cell[] output)
        {
            Console.WriteLine($"\n📥 [LEARNING] Input SDR: {string.Join(",", input.Take(20))}...");

            tm.Compute(input, learn: true);

            // 🔍 Debugging: Check if HTM is actually learning

            string key = string.Join(",", input);
            if (!trainingData.ContainsKey(key))
            {
                trainingData[key] = input;
            }
        }

        public int[] GetPredictedInputValue(Cell[] unclassifiedCells)
        {
            Console.WriteLine("\n🔍 [DEBUG] Entered GetPredictedInputValue()");

            if (unclassifiedCells == null || unclassifiedCells.Length == 0)
            {
                Console.WriteLine("⚠️ [WARNING] No unclassified cells provided!");
                return Array.Empty<int>();
            }

            int[] inputSdr = unclassifiedCells.Select(c => c.Index).ToArray();
            Console.WriteLine($"🟡 [INFO] Extracted Input SDR: {string.Join(",", inputSdr.Take(10))}...");

            var predictedResults = GetPredictedInputValues(inputSdr, 1);

            if (predictedResults == null || predictedResults.Count == 0)
            {
                Console.WriteLine("⚠️ [WARNING] No predictions found!");
                return Array.Empty<int>();
            }

            Console.WriteLine($"✅ [SUCCESS] Prediction generated.");
            return predictedResults[0].PredictedInput;
        }

        public List<ClassifierResult<int[]>> GetPredictedInputValues(int[] inputSdr, short howMany = 3)
        {
            tm.Compute(inputSdr, learn: false);

            var results = new List<ClassifierResult<int[]>>();

            foreach (var kv in trainingData)
            {
                int[] storedSdr = kv.Value;
                if (storedSdr.Length != inputSdr.Length)
                {
                    continue;
                }

                double similarity = ComputeHybridSimilarity(storedSdr, inputSdr);
                Console.WriteLine($"📊 [SIMILARITY] {similarity} for stored SDR: {string.Join(",", storedSdr.Take(10))}...");

                results.Add(new ClassifierResult<int[]>
                {
                    PredictedInput = storedSdr, // 🔥 Removed Unnecessary Bit Inversion
                    Similarity = similarity
                });
            }

            return results.OrderByDescending(r => r.Similarity).Take(howMany).ToList();
        }

        private static double ComputeHybridSimilarity(int[] a, int[] b)
        {
            double jaccard = MathHelpers.JaccardSimilarityofBinaryArrays(a, b);
            double cosineSim = 1.0 - CosineDistance(a, b);
            return (jaccard + cosineSim) / 2.0;
        }

        private static double CosineDistance(int[] a, int[] b)
        {
            double dot = a.Zip(b, (x, y) => x * y).Sum();
            double magA = Math.Sqrt(a.Sum(x => x * x));
            double magB = Math.Sqrt(b.Sum(y => y * y));
            return magA == 0 || magB == 0 ? 1.0 : 1 - (dot / (magA * magB));
        }
    }
}
