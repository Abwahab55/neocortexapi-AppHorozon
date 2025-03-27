using System.Collections.Generic;
using System.Linq;
using NeoCortexApi;
using NeoCortexApi.Classifiers;
using NeoCortexApi.Entities;

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
            {
                trainingData[hash] = input;
            }
        }

        public int[] GetPredictedInputValue(Cell[] cells) =>
            GetPredictedInputValues(cells.Select(c => c.Index).ToArray(), 1)
                .FirstOrDefault()?.PredictedInput;

        public List<ClassifierResult<int[]>> GetPredictedInputValues(int[] sdr, short n)
        {
            // Feed the SDR into TM to update internal state
            tm.Compute(sdr, learn: false);

            // Return the top-n closest matches from training data
            return trainingData.Values.Select(stored =>
                new ClassifierResult<int[]>
                {
                    PredictedInput = stored,
                    Similarity = stored.Zip(sdr, (a, b) => a == b ? 1 : 0).Sum()
                })
                .OrderByDescending(x => x.Similarity)
                .Take(n)
                .ToList();
        }
    }
}
