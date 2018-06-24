using System;
namespace GeneticAlgorithm
{
    public class Sample
    {
        public string Code;
        public int Value;
        public int Eval;

        public Sample()
        {
            Code = string.Empty;
            Value = 0;
            Eval = -1;
        }
    }
}
