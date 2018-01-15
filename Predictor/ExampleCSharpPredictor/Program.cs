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
                graph.Import(File.ReadAllBytes(@"C:\Users\Ben\Desktop\frozen.pb"));

                TFSession session = new TFSession(graph);
                TFSession.Runner runner = session.GetRunner();

                float[] x1 = new float[] { 239, 958, 8, 34, 239 };
                float[] x2 = new float[] { 239, 958, 8, 34, 239 };
                float[] x3 = new float[] { 239, 958, 8, 34, 239 };

                TFTensor x = new TFTensor(new float[][] { x1, x2, x3 });

                runner.AddInput(graph["Placeholder"][0], x);
                runner.Fetch(graph["add"][0]);

                var output = runner.Run();

                TFTensor result = output[0];

                var v = result.GetValue(); 

                return;
            }
        }
    }
}
