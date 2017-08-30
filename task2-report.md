# Task description
任務需求2：recommendation task 。

dataset 檔案名稱為rs.csv  (user：顾客／会员编号；item：商品编号；qty：交易量；datetime：交易时间)

請建模預測顧客購買行為（顧客-> List[商品]），並自行切分 training、validation、test三个dataset，或是以Cross Validation等方式進行驗證。另外，請說明驗證方式與驗證結果。

完成任務後寄回：

1:測試結果報告。 

2:可把code git上去git hub，提供git link給我們參考。

# Workflow
1. Define training and testing dataset
2. Run collaborative filtering
3. Define and calculate performance metrics

# Define training and testing dataset
Data preprocessing
```
val file = spark.read
    .option("delimiter", ",")
    .option("header", "true")
    .csv("rs.csv")

import java.sql.Timestamp  

val dataset = file.select("user", "item", "qty", "datetime").as[(String, Long, Int, Timestamp)]

import org.apache.spark.ml.feature.StringIndexer

val indexer = new StringIndexer()
  .setInputCol("user")
  .setOutputCol("userIndex")

val indexed = indexer.fit(dataset).transform(dataset).select("userIndex", "item", "qty", "datetime").as[(Double, Long, Int, Timestamp)]
```
Set last order for each customer as test dataset
```
import org.joda.time.DateTime

case class Entry(user: Int, item: Long, quantity: Int,dt: org.joda.time.DateTime)
val ds = indexed.rdd.map(a => Entry(a._1.toInt, a._2, a._3, new DateTime(a._4)))

val testset = ds.groupBy(a => a.user).map(a => (a._1, a._2.toList.groupBy(a => a.dt).toList.sortWith((x, y) => x._1.isBefore(y._1)).last))
val trainset = ds.groupBy(a => a.user).map(a => (a._1, a._2.toList.groupBy(a => a.dt).toList.sortWith((x, y) => x._1.isBefore(y._1)).dropRight(1)))

val trainSet = trainset.map(a => (a._1, a._2.map(b => b._2).reduceLeft((x, y) => x ++ y)))
val testSet = testset.map(a => (a._1, a._2._2))
```

# Collaborative Filtering
```
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating

val ratings = trainSet.flatMap(a => a._2.map(b => Rating(a._1.toInt, b.item.toInt, b.quantity.toDouble)))
val rank = 10
val numIterations = 10
val model = ALS.train(ratings, rank, numIterations, 0.01)

```

# Performance metrics (驗證方式)


答案是每個顧客最後一筆訂單的商品(List[商品]), Collaborative filtering 會推薦10樣商品給每個人, 兩個集合可以計算precision, recall, f-score, 最後平均f-score為驗證結果


```
def precision(rec: List[Int], answer: List[Int]) = rec.intersect(answer).size.toDouble / rec.size
def recall(rec: List[Int], answer: List[Int]) = rec.intersect(answer).size.toDouble / answer.size

def fscore(rec: List[Int], answer: List[Int]) = {
    if (precision(rec, answer) == 0) 0
    else 2 * precision(rec, answer) * recall(rec, answer) / (precision(rec, answer) + recall(rec, answer))
}

val recommendation = model.recommendProductsForUsers(10).map(a => (a._1, a._2.map(b => b.product).toList))
val answer = testSet.map(a => (a._1, a._2.map(b => b.item.toInt)))

val fScore = recommendation.join(answer).map(a => fscore(a._2._1, a._2._2)).sum
val userCount = recommendation.join(answer).map(a => fscore(a._2._1, a._2._2)).count
fScore/userCount
```
# Result: mean f-score = 0.0046