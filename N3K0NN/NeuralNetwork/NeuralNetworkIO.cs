using System;
using System.IO;
using System.Xml.Serialization;

namespace N3K0NN.NeuralNetwork
{
    public static class NeuralNetworkIo
    {
        public static void Save(NN nn, string path)
        {
            var    conf = nn.GetConfiguration();
            var    xml  = new XmlSerializer(typeof(NnConfiguration));

            using var fs = new FileStream(path, FileMode.OpenOrCreate);
            xml.Serialize(fs, conf);
        }

        public static NnConfiguration Load(string path)
        {
            var xml = new XmlSerializer(typeof(NnConfiguration));
            using var fs = new FileStream(path, FileMode.OpenOrCreate);
            return (NnConfiguration) xml.Deserialize(fs);
        }
    }
}