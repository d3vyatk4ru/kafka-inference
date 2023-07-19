import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.functions.{col, collect_list, explode, regexp_replace, split, udf}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{CountVectorizer, IndexToString, StringIndexer}
import org.apache.spark.ml.{Pipeline, PipelineModel}

class train {

  def main(args : Array[String]): Unit = {

    // set params
    val conf = new SparkConf()
      .set("spark.train_dataset", "/labs/laba07/laba07.json")
      .set("spark.pipеline", "logreg_pipeline")

    // start spark session
    val spark = SparkSession
      .builder
      .appName("lab04_9kin")
      .getOrCreate()

    // read dataframe
    val df: DataFrame = spark
      .read
      .json(conf.get("spark.train_dataset"))

    // read necessary cols
    val dataset: DataFrame = df.select(
      col("uid"),
      col("gender_age"),
      col("visits.url")
    )

    // prepare dataset for log reg
    val prepairedDataset: DataFrame = dataset
      .withColumn("url", explode(col("url")))
      .withColumn("url", split(col("url"), "/").getItem(2))
      .withColumn("url", regexp_replace(col("url"), "www.", ""))
      .groupBy(col("uid"), col("gender_age"))
      .agg(collect_list(col("url")).alias("domains"))

    // create features and labels for training
    val cv = new CountVectorizer()
      .setInputCol("domains")
      .setOutputCol("features")

    val indexer = new StringIndexer()
      .setInputCol("gender_age")
      .setOutputCol("label")

    val labels = indexer.fit(prepairedDataset).labels

    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.001)

    // reverse transform from index to age for test
    val transform2Age = new IndexToString()
      .setInputCol("prediction")
      .setLabels(labels)
      .setOutputCol("result")

    val pipeline = new Pipeline()
      .setStages(Array(cv, indexer, lr, transform2Age))

    val model = pipeline.fit(prepairedDataset)

    model
      .write
      .overwrite()
      .save(conf.get("spark.pipеline"))

    spark.stop()

  }
}
