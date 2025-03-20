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
        private SpatialPooler sp;
        private TemporalMemory tm;
        private Connections connections;

        public HtmImageClassifier()
        {
            // Initialize SP/TM with 28x28 columns (784 columns).
            connections = new Connections(new HtmConfig
            {
                ColumnDimensions = new int[] { 28, 28 },
                InputDimensions = new int[] { 28, 28 },
                NumInputs = 784,
                PotentialPct = 0.6,
                SynPermInactiveDec = 0.005,
                SynPermActiveInc = 0.03,
                SynPermConnected = 0.2,
                NumActiveColumnsPerInhArea = 40
            });

            sp = new SpatialPooler();
            sp.Init(connections);

            tm = new TemporalMemory();
            tm.Init(connections);
        }

        public void Learn(int[] input, Cell[] output)
        {
            int[] activeColumns = new int[connections.HtmConfig.NumColumns];

            // Compute spatial pooler output (active columns)
            sp.compute(input, activeColumns, learn: true);
            tm.Compute(activeColumns, learn: true);

            // Store trained patterns
            string key = string.Join(",", input);
            if (!trainingData.ContainsKey(key))
            {
                trainingData[key] = activeColumns;
            }
        }

        // ? Implementing the required method: GetPredictedInputValue(Cell[])
        public int[] GetPredictedInputValue(Cell[] unclassifiedCells)
        {
            if (unclassifiedCells == null || unclassifiedCells.Length == 0)
                return Array.Empty<int>();

            // Convert Cell[] to SDR indices
            int[] inputSdr = unclassifiedCells.Select(c => c.Index).ToArray();

            var predictedResults = GetPredictedInputValues(inputSdr, 1);

            if (predictedResults.Count > 0)
                return predictedResults[0].PredictedInput;

            return Array.Empty<int>();
        }

        public List<ClassifierResult<int[]>> GetPredictedInputValues(int[] inputSdr, short howMany = 3)
        {
            int[] activeColumns = new int[connections.HtmConfig.NumColumns];
            sp.compute(inputSdr, activeColumns, learn: false);
            tm.Compute(activeColumns, learn: false);

            var results = new List<ClassifierResult<int[]>>();

            foreach (var kv in trainingData)
            {
                string key = kv.Key;
                int[] storedSdr = kv.Value;

                double similarity = ComputeHybridSimilarity(storedSdr, activeColumns);
                results.Add(new ClassifierResult<int[]>
                {
                    PredictedInput = storedSdr,
                    NumOfSameBits = storedSdr.Intersect(activeColumns).Count(),
                    Similarity = similarity
                });
            }

            results = results.OrderByDescending(r => r.Similarity).Take(howMany).ToList();

            Console.WriteLine("----- HTM Classifier Debug -----");
            foreach (var res in results)
            {
                Console.WriteLine($"Similarity: {res.Similarity * 100:0.00}%");
            }
            Console.WriteLine("-------------------------------");

            return results;
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
