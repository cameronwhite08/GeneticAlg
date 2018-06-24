using System;
namespace GeneticAlgorithmChar
{
    public class Sample
    {
        public int[] Code;
        public long Eval;

        public Sample(int codeLength)
        {
            Code = new int[codeLength];
            Eval = -1;
        }
    }
}
