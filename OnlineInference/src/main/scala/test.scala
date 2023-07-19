import breeze.linalg.split
import org.apache.spark.SparkConf
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, collect_list, explode, from_json, regexp_replace}
import org.apache.spark.sql.types.{ArrayType, LongType, StringType, StructType}
import org.apache.spark.sql.streaming.Trigger

class test {

  val schema = new StructType()
    .add("uid", StringType)
    .add("visits", ArrayType(
      new StructType()
        .add("url", StringType)
        .add("timestamp", LongType)
    ))


  def main(args: Array[String]): Unit = {

    // set params
    val conf = new SparkConf()
      .set("spark.pipеline", "logreg_pipeline")
      .set("spark.data_for_predict_topic", "daniil_devyatkin")
      .set("spark.output_topic", "daniil_devyatkin_lab07_out")

    // start spark session
    val spark = SparkSession
      .builder
      .appName("lab04_9kin_test")
      .getOrCreate();


    val df = spark
      .readStream
      .format("kafka")
      .option("kafka.bootstrap.servers", "spark-master-1:6667")
      .option("subscribe", conf.get("spark.data_for_predict_topic"))
      .option("includeHeaders", "true")
      .load()
      .select(col("key").cast("string"), col("value").cast("string"))
      .withColumn("value", from_json(col("value").cast("string"), schema))
      .select(col("value.uid"), col("value.visits.url"))
      .withColumn("url", explode(col("url")))
      .withColumn("url", split(col("url"), "/").getItem(2))
      .withColumn("url", regexp_replace(col("url"), "www.", ""))
      .groupBy(col("uid"))
      .agg(collect_list(col("url")).alias("domains"))

    val model = PipelineModel.load(conf.get("spark.pipеline"))

    val predict: DataFrame = model
      .transform(df)
      .withColumnRenamed("result", "gender_age")
      .select(
        col("uid"),
        col("gender_age")
      )

    val query = predict
      .selectExpr("CAST(uid AS STRING) AS key", "to_json(struct(*)) AS value")
      .writeStream
      .trigger(Trigger.ProcessingTime("5 seconds"))
      .format("kafka")
      .option("checkpointLocation", "lab07-chk")
      .option("kafka.bootstrap.servers", "10.0.0.5:6667")
      .option("topic", conf.get("spark.output_topic"))
      .option("maxOffsetsPerTrigger", 200)
      .outputMode("update")
      .start

    spark.stop()
  }
}
