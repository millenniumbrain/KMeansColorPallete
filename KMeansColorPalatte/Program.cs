using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Newtonsoft.Json;
using OpenCvSharp;
using Accord.MachineLearning;
using Accord.Math.Distances;

namespace KMeansColorPalatte
{
    class Program
    {
        private class Color 
        {
            public byte R { get; set; }
            public byte G { get; set; }
            public byte B { get; set; }
        }
        static void Main(string[] args)
        {
            Task.Run(async () => await ConvertImage()).Wait();
        }

        private static async Task ConvertImage()
        {

            var imageData = await ReadImgFileAsync("test-2.jpg");
            var json = GenerateColorPalette(imageData, 5);



            Console.WriteLine(json);
            Console.ReadLine();
        }

        private static string GenerateColorPalette(byte[] imgData, int numClusters)
        {
            var img = Cv2.ImDecode(imgData, ImreadModes.Color);

            var colors = new List<Vec3b>();
            var mat3 = new Mat<Vec3b>(img);
            var indexer = mat3.GetIndexer();
            for (int y = 0; y < img.Height / 10; y++)
            {
                for (int x = 0; x < img.Width / 10; x++)
                {
                    var color = indexer[y, x];
                    byte temp = color.Item0;
                    color.Item0 = color.Item2;
                    color.Item2 = temp;
                    indexer[y, x] = color;
                    colors.Add(indexer[y, x]);
                }
            }

            var observations = colors.Select(c => new double[] { c.Item0, c.Item1, c.Item2 }).ToArray();
            var kmeans = new KMeans(numClusters, new SquareEuclidean())
            {
                Tolerance = 0.05
            };
            var clusters = kmeans.Learn(observations);

            var computedColors = clusters.Select(c => new Color { R = (byte)Math.Round(c.Centroid[0], 0), G = (byte)Math.Round(c.Centroid[1], 0), B = (byte)Math.Round(c.Centroid[2], 0) }).ToList();

            string json = JsonConvert.SerializeObject(computedColors, Formatting.Indented);

            return json;
        }

        private static async Task<byte[]> ReadImgFileAsync(string filename) {
            byte[] imageBuffer;

            using (var stream = new FileStream(filename, FileMode.Open))
            {
                var bufferLength = (int)stream.Length;
                imageBuffer = new byte[bufferLength];
                var totalBytesRead = 0;
                while (totalBytesRead < imageBuffer.Length)
                {
                    totalBytesRead += await stream.ReadAsync(imageBuffer, totalBytesRead, bufferLength - totalBytesRead);
                }
            }
            return imageBuffer;
        }
    }
}
