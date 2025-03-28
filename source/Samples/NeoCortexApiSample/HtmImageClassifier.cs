using System;
using System.Collections.Generic;
using System.Linq;
using NeoCortexApi;
using NeoCortexApi.Classifiers;
using NeoCortexApi.Entities;

namespace NeoCortexApiSample
{
    /// <summary>
    /// HTM-based image classifier that learns a sequence of SDRs and predicts future inputs
    /// using Temporal Memory's predictive cells.
    /// </summary>
    public class HtmImageClassifier : IClassifier<int[], string>
    {
        private readonly List<int[]> learnedSequence = new(); // Stores the input SDRs during training
        private readonly TemporalMemory tm;
        private readonly Connections connections;
        private readonly int width, height;

        public HtmImageClassifier(int width = 64, int height = 64)
        {
            this.width = width;
            this.height = height;

            var config = new HtmConfig
            {
                ColumnDimensions = new[] { width, height },
                InputDimensions = new[] { width, height },
                CellsPerColumn = 8,
                NumInputs = width * height,
                PotentialPct = 0.6,
                SynPermConnected = 0.2
            };

            connections = new Connections(config);
            tm = new TemporalMemory();
            tm.Init(connections);
        }

        /// <summary>
        /// Learns the given input SDR by feeding it into the Temporal Memory.
        /// Stores the input for future reconstruction comparison.
        /// </summary>
        public void Learn(int[] input, Cell[] _)
        {
            tm.Compute(input, learn: true);
            learnedSequence.Add(input);
        }

        /// <summary>
        /// Resets the internal state of Temporal Memory.
        /// This is usually done between training cycles.
        /// </summary>
        public void ResetTemporalMemory()
        {
            tm.Reset(connections);
        }

        /// <summary>
        /// Returns the best predicted input SDR based on the current HTM predictive state.
        /// </summary>
        public int[] GetPredictedInputValue(Cell[] _)
        {
            return GetPredictedInputValues(Array.Empty<int>(), 1)
                .FirstOrDefault()?.PredictedInput;
        }

        /// <summary>
        /// Computes the predicted SDR by activating the HTM and selecting the closest match
        /// from previously learned inputs based on overlap with predictive cells.
        /// </summary>
        public List<ClassifierResult<int[]>> GetPredictedInputValues(int[] sdr, short n)
        {
            if (sdr.Length > 0)
                tm.Compute(sdr, learn: false);

            // Get all predictive cells from the TM state
            var predictiveCells = tm.GetPredictiveCells();

            // Convert predictive cells into column indices
            var predictedColumns = predictiveCells
                .Select(cellIdx => connections.Cells[cellIdx].ParentColumnIndex)
                .Distinct()
                .ToHashSet();

            // Reconstruct predicted SDR from columns
            int[] predictedSdr = new int[width * height];
            foreach (int colIdx in predictedColumns)
            {
                if (colIdx >= 0 && colIdx < predictedSdr.Length)
                    predictedSdr[colIdx] = 1;
            }

            // Select the most similar SDR from training
            int[] bestMatch = predictedSdr;
            double bestScore = double.MinValue;

            foreach (var stored in learnedSequence)
            {
                double overlap = ComputeOverlap(predictedSdr, stored);
                if (overlap > bestScore)
                {
                    bestScore = overlap;
                    bestMatch = stored;
                }
            }

            return new List<ClassifierResult<int[]>>
            {
                new ClassifierResult<int[]>
                {
                    PredictedInput = bestMatch,
                    Similarity = bestScore
                }
            };
        }

        /// <summary>
        /// Returns the number of overlapping bits (1s) between two SDRs.
        /// </summary>
        private double ComputeOverlap(int[] a, int[] b)
        {
            return a.Zip(b, (x, y) => x & y).Sum();
        }
    }
}
