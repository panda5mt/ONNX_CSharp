using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Microsoft.ML.Trainers;

// データクラスの定義
public class ModelInput
{
    [LoadColumn(0)]
    public float Feature1;

    [LoadColumn(1)]
    public float Feature2;

    [LoadColumn(2)]
    public float Feature3;

    [LoadColumn(3)]
    public float Feature4;

    [ColumnName("Label")]
    [LoadColumn(4)]
    public float Label;
}

public class ModelOutput
{
    [ColumnName("PredictedLabel")]
    public bool Prediction { get; set; }

    public float Probability { get; set; }

    public float Score { get; set; }
}

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

    static void Learning(string? csvPath, string? onnxPath)
    {
        // MLContextの作成
        var mlContext = new MLContext();
        IDataView? dataView;

        // データのロード
        try
        {
            dataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                            path: csvPath,
                            hasHeader: true,
                            separatorChar: ',');

        }
        catch (Exception ex)
        {
            Console.WriteLine(ex.Message);
            return;
        }

        // 学習パイプラインの定義
        var trainingDataView = dataView;
        var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
            .Append(mlContext.Transforms.Concatenate("Features", "Feature1", "Feature2", "Feature3", "Feature4"))
            .Append(mlContext.MulticlassClassification.Trainers.LightGbm(labelColumnName: "Label", featureColumnName: "Features"))
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

        // モデルの学習
        var model = pipeline.Fit(trainingDataView);

        // モデルをONNX形式でエクスポート
        if (onnxPath != null)
        {
            using var stream = File.Create(onnxPath);
            mlContext.Model.ConvertToOnnx(model, trainingDataView, stream);
        }
        else {
            return; 
        }

        Console.WriteLine("Learning end.");
        

    }
    static void Predict(string? onnxPath)
    {
        // ONNXモデルのパス
        //string modelPath = "random_forest.onnx";

        // ONNXランタイムのセッションを作成
        InferenceSession? session;
        try
        {
            session = new InferenceSession(onnxPath);
        }catch (Exception ex)
        {
            Console.WriteLine(ex.Message);
            return;
        }

        /*
        // ONNXの中身確認用
        using var mySession = new InferenceSession(onnxPath);
        foreach (var input in mySession.InputMetadata)
        {
            Console.WriteLine($"Input Name: {input.Key}");
            Console.WriteLine($"Input Type: {input.Value.ElementType}");
            Console.WriteLine($"Input Shape: {string.Join(", ", input.Value.Dimensions)}");
        }
        */

        // 入力データ
        float[] feature1 = new float[] { 3.0f };
        float[] feature2 = new float[] { 4.0f };
        float[] feature3 = new float[] { 5.0f };
        float[] feature4 = new float[] { 6.0f };
        float[] dummyLabel = new float[] { 0 }; // ダミーのラベル値

        // 各特徴量とダミーラベルをテンソルに変換
        var tensorFeature1 = new DenseTensor<float>(feature1, new[] { 1, 1 });
        var tensorFeature2 = new DenseTensor<float>(feature2, new[] { 1, 1 });
        var tensorFeature3 = new DenseTensor<float>(feature3, new[] { 1, 1 });
        var tensorFeature4 = new DenseTensor<float>(feature4, new[] { 1, 1 });
        var tensorDummyLabel = new DenseTensor<float>(dummyLabel, new[] { 1, 1 }); // ダミーラベルのテンソル

        // 推論を実行
        var inputs = new List<NamedOnnxValue>{
            NamedOnnxValue.CreateFromTensor("Feature1", tensorFeature1),
            NamedOnnxValue.CreateFromTensor("Feature2", tensorFeature2),
            NamedOnnxValue.CreateFromTensor("Feature3", tensorFeature3),
            NamedOnnxValue.CreateFromTensor("Feature4", tensorFeature4),
            NamedOnnxValue.CreateFromTensor("Label", tensorDummyLabel) // ダミーラベルを含める
        };

        // 推論を実行
        using var results = session.Run(inputs);

        foreach (var result in results)
        {
            Console.WriteLine($"Name: {result.Name}");
            if (result.Value is Tensor<float> tensorValue)
            {
                // Tensor型の結果を表示
                Console.WriteLine($"Values: {string.Join(", ", tensorValue.ToArray())}");
            }
            else
            {
                // Tensor型以外の結果を表示
                Console.WriteLine("Non-tensor type");
            }
        }


    }
}

