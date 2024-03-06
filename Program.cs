using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Microsoft.ML.Trainers.LightGbm;


/*
 *Pi系のSBC(linux-arm64)はLightGBMを自前でビルドする。
 このプロジェクトでNuGetから導入したML.netが使用しているLightGBMはv3系なので、v3.0.0を使用した。
 git clone --recursive https://github.com/microsoft/LightGBM -b v3.0.0 --depth 1
 mkdir LightGBM/build/
 cd LightGBM/build/
 cmake ..
 make -j4
 cp /path/to/LightGBM/lib_lightgbm.so /path/to/dotnet_proj/runtimes/linux-arm64/native/

 ビルド後、このプロジェクト起動してlib_lightgbm.soがないというエラー以外、特に
 The type initializer for 'Microsoft.ML.OnnxRuntime.NativeMethods' threw an exception.
 みたいなメッセージが出たら下記を実行

dotnet add package Microsoft.ML
dotnet add package Microsoft.ML.OnnxRuntime
dotnet add package Microsoft.ML.OnnxTransformer
*/

class Program
{
    static void Main(/*string[] args*/)
    {
        string? onnxPath, csvPath;
        string? csvName = "record.csv";
     
        string pwd_folder = System.Environment.CurrentDirectory;

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows) && Debugger.IsAttached) // Visual Studioでデバッグ中か判断
        {
            onnxPath = Path.Combine(pwd_folder, "../../../model.onnx");
            csvPath = Path.Combine(pwd_folder, "../../../" + csvName);

        }
        else
        {
            onnxPath = Path.Combine(pwd_folder, "model.onnx");
            csvPath = Path.Combine(pwd_folder, csvName);
        }

        Learning(csvPath, onnxPath);
        Predict(onnxPath);


    }
    
    static int columnCount;
    static void Learning(string? csvPath, string? onnxPath)
    {
        var mlContext = new MLContext();
        IDataView? dataView;
        if (File.Exists(csvPath))
        {
            // CSVファイルの最初の行を読み込み、列数を取得
            string firstLine = File.ReadLines(csvPath).First();
            columnCount = firstLine.Split(',').Length; // カンマで分割して列数を取得

            // TextLoaderを動的に設定
            var textLoaderColumns = new TextLoader.Column[columnCount];
            for (int i = 0; i < columnCount - 1; i++)
            {
                textLoaderColumns[i] = new TextLoader.Column($"Feature{i}", DataKind.Single, i);
            }
            // 最後の列をラベルとして設定
            textLoaderColumns[columnCount - 1] = new TextLoader.Column("Label", DataKind.Single, columnCount - 1);

            var textLoader = mlContext.Data.CreateTextLoader(new TextLoader.Options
            {
                Columns = textLoaderColumns,
                HasHeader = true,
                Separators = new[] { ',' }
            });

            // データのロード
            try
            {
                dataView = textLoader.Load(csvPath);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                return;
            }

            var options = new LightGbmMulticlassTrainer.Options
            {
                LabelColumnName = "Label",
                FeatureColumnName = "Features",
                MinimumExampleCountPerLeaf = 5,
                NumberOfLeaves = 31,
                NumberOfIterations = 100,
                LearningRate = 0.1,
                MaximumBinCountPerFeature = 50,
            };

            // 学習パイプラインの定義
            // 特徴量列名を動的に生成
            var featureColumnNames = textLoaderColumns.Take(columnCount - 1).Select(col => col.Name).ToArray();
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.Transforms.Concatenate("Features", featureColumnNames))
                .Append(mlContext.MulticlassClassification.Trainers.LightGbm(options))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // モデルの学習
            var model = pipeline.Fit(dataView);

            // モデルをONNX形式でエクスポート
            if (onnxPath != null)
            {
                using var stream = File.Create(onnxPath);
                mlContext.Model.ConvertToOnnx(model, dataView, stream);
            }
            else
            {
                return;
            }

            Console.WriteLine("Learning end.");
        }
        else
        {
            Console.WriteLine("No CSV FILE!!!");
        }
    }

    static void PrintModelInputNames(string modelPath)
    {
        using var session = new InferenceSession(modelPath);
        foreach (var inputMeta in session.InputMetadata)
        {
            Console.WriteLine($"Input Name: {inputMeta.Key}");
        }
    }
    static void Predict(string? onnxPath)
    {
        // ONNXランタイムのセッションを作成
        InferenceSession session;
        try
        {
            session = new InferenceSession(onnxPath);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error creating InferenceSession: {ex.Message}");
            return;
        }

        // 特徴量のテンソルを動的に作成してリストに追加
        var inputs = new List<NamedOnnxValue>();
        for (int i = 0; i < columnCount - 1; i++)
        {
            // 特徴量名を"FeatureX"の形式で生成
            string featureName = $"Feature{i}";
            inputs.Add(NamedOnnxValue.CreateFromTensor(featureName, new DenseTensor<float>(new float[] { /*featureValues[i]*/i+4 }, new[] { 1, 1 })));
        }

        // ダミーラベルの値（ここでは0を使用していますが、必要に応じて変更してください）
        float[] dummyLabelValue = { 0 };
        inputs.Add(NamedOnnxValue.CreateFromTensor("Label", new DenseTensor<float>(dummyLabelValue, new[] { 1, 1 })));

        // 推論を実行
        using var results = session.Run(inputs);

        // 結果の出力
        foreach (var result in results)
        {
            Console.WriteLine($"Name: {result.Name}");
            if (result.Value is Tensor<float> tensorValue)
            {
                Console.WriteLine($"Values: {string.Join(", ", tensorValue.ToArray())}");
            }
            else
            {
                Console.WriteLine("Non-tensor type");
            }
        }
    }
}

