# Update outline
1. Add bigram feature
2. Use TFIDF to scale the features
3. Use random forest classifier

# Updated feature

Extract bigram feature
```
import org.apache.spark.ml.feature.NGram

val ngram = new NGram().setN(2).setInputCol("words").setOutputCol("ngrams")

val ngramDataFrame = ngram.transform(seged)

ngramDataFrame.show
```

Calculate Term frequency
```
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}

val cvModel: CountVectorizerModel = new CountVectorizer()
  .setInputCol("words")
  .setOutputCol("rawFeatures1")
  .setMinDF(4)
  .fit(ngramDataFrame)
  
val data = cvModel.transform(ngramDataFrame)
  
val cvModel2: CountVectorizerModel = new CountVectorizer()
  .setInputCol("ngrams")
  .setOutputCol("rawFeatures2")
  .setMinDF(4)
  .fit(data)
  

val data2 = cvModel2.transform(data)
data2.show
```

Calculate inverse document frequency
```
/* TFIDF  */
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}

//val hashingTF = new HashingTF()
//  .setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(1370)

//val featurizedData = hashingTF.transform(seged)

//featurizedData.show

val idf1 = new IDF().setInputCol("rawFeatures1").setOutputCol("features1")
val idfModel1 = idf1.fit(data2)

val rescaledData1 = idfModel1.transform(data2)

val idf2 = new IDF().setInputCol("rawFeatures2").setOutputCol("features2")
val idfModel2 = idf2.fit(rescaledData1)

val rescaledData2 = idfModel2.transform(rescaledData1)
rescaledData2.show
```
Combine unigram and bigram
```
import org.apache.spark.ml.feature.VectorAssembler

val assembler = new VectorAssembler()
  .setInputCols(Array("features1", "features2"))
  .setOutputCol("features")

val traingingData = assembler.transform(rescaledData2.select("label", "features1", "features2"))
traingingData.show
```
# Result of adding bigram and tfidf: acurracy = 0.6596

# Use Random Forest
```
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}

// Train a RandomForest model.
val rf = new RandomForestClassifier()
  .setLabelCol("label")
  .setFeaturesCol("features")
  .setThresholds(Array(0.012734584450402145,0.06735924932975872, 0.5318364611260054, 0.19168900804289543, 0.19638069705093833, 0.1))
  .setNumTrees(200)

val model = rf.fit(trData)

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

val predictions = model.transform(teData)

val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
```

# Result of using random forest: accuracy = 0.59