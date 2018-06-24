using System;
namespace GeneticAlgorithmBinary
{
    public class Sample
    {
        public string Code;
        public long Value;
        public long Eval;

        public Sample()
        {
            Code = string.Empty;
            Value = 0;
            Eval = -1;
        }
    }
}
