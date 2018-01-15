using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorFlow;

namespace ExampleCSharpPredictor
{
    class Program
    {
        static void Main(string[] args)
        {
            using (TFGraph graph = new TFGraph())
            {
                graph.Import(File.ReadAllBytes(@"C:\Users\Ben\Desktop\Training\model.ckpt.meta"));

                return; 
            }
        }
    }
}
