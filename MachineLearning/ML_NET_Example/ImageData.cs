﻿using Microsoft.ML.Data;

namespace ML_NET_Example
{
    public class ImageData
    {
        [LoadColumn(0)]
        public byte[] Image { get; set; }

        [LoadColumn(1)]
        public string Label { get; set; }
    }
}
