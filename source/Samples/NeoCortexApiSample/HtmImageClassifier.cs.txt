using System;
using System.Collections.Generic;
using System.Linq;
using NeoCortexApi;
using NeoCortexApi.Classifiers;
using NeoCortexApi.Entities;

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
                SynPermActiveInc = 0.05,
                SynPermConnected = 0.2,
                NumActiveColumnsPerInhArea = 50
            });
            tm = new TemporalMemory();
            tm.Init(connections);
        }

        public void Learn(int[] input, Cell[] output)
        {
            tm.Compute(input, learn: true);
            string key = string.Join(",", input);
            if (!trainingData.ContainsKey(key))
            {
                trainingData[key] = input;
            }
        }

        public int[] GetPredictedInputValue(Cell[] unclassifiedCells)
        {
            if (unclassifiedCells == null || unclassifiedCells.Length == 0)
            {
                return Array.Empty<int>();
            }

            int[] inputSdr = unclassifiedCells.Select(c => c.Index).ToArray();
            var predictedResults = GetPredictedInputValues(inputSdr, 1);

            return predictedResults.Count > 0 ? predictedResults[0].PredictedInput : Array.Empty<int>();
        }

        public List<ClassifierResult<int[]>> GetPredictedInputValues(int[] inputSdr, short howMany = 3)
        {
            tm.Compute(inputSdr, learn: false);
            var results = new List<ClassifierResult<int[]>>();

            foreach (var kv in trainingData)
            {
                int[] storedSdr = kv.Value;
                if (storedSdr.Length != inputSdr.Length)
                    continue;

                results.Add(new ClassifierResult<int[]>
                {
                    PredictedInput = storedSdr,
                    Similarity = 0 // Placeholder since similarity is calculated in Program.cs
                });
            }
            return results.OrderByDescending(r => r.Similarity).Take(howMany).ToList();
        }
    }
}
