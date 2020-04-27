using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace ML_NET_Example
{
    public class ImagePrediction
    {
        [ColumnName("Score")]
        public float[] Score { get; set; }

        [ColumnName("PredictedLabel")]
        public string PredictedLabel { get; set; }
    }
}
