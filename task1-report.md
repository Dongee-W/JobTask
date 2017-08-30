# Task description
任務需求1：做一個情緒預測模型。 

dataset 檔案名稱為Ch_trainfile_Sentiment_3000.txt

請使用svm進行情緒預測模型。另外，請再想任何一個模型要能比svm好，用相同資料當驗證。

input：句子 or 一個段落

output：-2, -1, 0, 1, 2

建議採用5-fold cross validation，回覆數據是多少。

如：用SVM做完5-fold cross validation後的精準度為80.12％

# Workflow
1. Text Segmentation
2. Bag of words representation
3. Train SVM classifier
4. Train Naive Bayes classifier

# Preprocessing and text segmentation
Parsing the text file
```
val file = sc.textFile("Ch_trainfile_Sentiment_3000.txt")

val sepChar1 = file.take(1)(0)(1)
val sepChar2 = file.take(1)(0)(2).toString

val parsed = file.map(a => a.replace(sepChar2, "").split(sepChar1).toList)
```

Chinese text segmentation with ansj_seg
```
import scala.collection.JavaConversions._

val seged = parsed.filter(a => a(0) != "").map{
    a => 
      import org.ansj.splitWord.analysis.ToAnalysis
  (a(0).toInt + 2, ToAnalysis.parse(a(1)).getTerms.toList.toArray.map(b => b.getRealName()))
    
}.toDF("label", "words")
```

# Bag of word representation of the text
```
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}

val cvModel: CountVectorizerModel = new CountVectorizer()
  .setInputCol("words")
  .setOutputCol("features")
  .setMinDF(3)
  .fit(seged)

val data = cvModel.transform(seged).drop("words")
``` 

# Training SVM model
```
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

val partition = data.randomSplit(Array(0.2, 0.2, 0.2, 0.2, 0.2))

val batch1 = partition(0).union(partition(1)).union(partition(2)).union(partition(3))
val batch2 = partition(0).union(partition(1)).union(partition(2)).union(partition(4))
val batch3 = partition(0).union(partition(1)).union(partition(3)).union(partition(4))
val batch4 = partition(0).union(partition(2)).union(partition(3)).union(partition(4))
val batch5 = partition(1).union(partition(2)).union(partition(3)).union(partition(4))

val lsvc = new LinearSVC()
  .setMaxIter(10)
  .setRegParam(0.1)

// instantiate the One Vs Rest Classifier.
val ovr = new OneVsRest().setClassifier(lsvc)
```

Cross validation fold 1 accuracy = 0.6515912897822446
```
// train the multiclass model.
val ovrModel = ovr.fit(batch1)

// score the model on test data.
val predictions = ovrModel.transform(partition(4))

// obtain evaluator.
val evaluator = new MulticlassClassificationEvaluator()
  .setMetricName("accuracy")

// compute the classification error on test data.
val accuracy = evaluator.evaluate(predictions)
```
Cross validation fold 2 accuracy = 0.6832191780821918

Cross validation fold 3 accuracy = 0.6532258064516129

Cross validation fold 4 accuracy = 0.6391231028667791

Cross validation fold 5 accuracy = 0.6795952782462057

# SVM result: accuracy = 0.66

# Train NaiveBayes model
```
import org.apache.spark.ml.classification.NaiveBayes

// Train a NaiveBayes model.
val model = new NaiveBayes()
  .fit(batch1)

// Select example rows to display.
val predictions = model.transform(partition(4))


val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
```
Cross validation fold 1 accuracy = 0.6532258064516129

Cross validation fold 2 accuracy = 0.6524822695035462

Cross validation fold 3 accuracy = 0.6465798045602605

Cross validation fold 4 accuracy = 0.680577849117175

Cross validation fold 5 accuracy = 0.6431095406360424

# Naive Bayes classifier result: accuracy = 0.655