import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import java.io._

object Isaac_Chien_task1 {
  def main(args:Array[String]) {
    val t1 = System.nanoTime

    // Load and parse the data
    val conf = new SparkConf()
    conf.setMaster("local")
    conf.setAppName("CF")
    val sc = new SparkContext(conf)


    val data = sc.textFile(args(0))
    val ratings = data
      .filter(line => line != "userId,movieId,rating,timestamp")
      .map(line => line.split(","))
      .map(word => Rating(word(0).toInt, word(1).toInt, word(2).toDouble))

    // Build the recommendation model using ALS
    val rank = 10
    val numIterations = 10
    val model = ALS.train(ratings, rank, numIterations, 0.01)


    var testFile = sc.textFile(args(1))
    var usersMovies = testFile
      .filter(line => line != "userId,movieId")
      .map(line => line.split(","))
      .map(word => (word(0).toInt, word(1).toInt))

    // Evaluate the model on rating data


    val predictions =
      model.predict(usersMovies).map { case Rating(user, product, rate) =>
        ((user, product), rate)
      }
//      .saveAsTextFile("Isaac_Chien_result_task_1.txt")

    val ratesAndPreds = ratings.map { case Rating(user, product, rate) =>
      ((user, product), rate)
    }.join(predictions)


    val diffMap = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
      val absDiff = Math.floor(Math.abs(r1 - r2));
      if (absDiff >= 4){
        (4, 1)
      } else {
        (absDiff.toInt, 1)
      }
    }.reduceByKey((a, b) => a + b)
      .collectAsMap()

    val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
      val err = (r1 - r2)
      err * err
    }.mean()

    val header = sc.parallelize(Seq("UserId,MovieId,Pred_rating"));
    val results = predictions
      .sortByKey()
//      .collect()
      .map(line => (line._1._1 + "," + line._1._2 + "," +line._2))

    (header ++ results).coalesce(1).saveAsTextFile("Isaac_Chien_result_task1.txt");

    for( key <- 0 to 3) {
      if (diffMap.exists(_._1 == key)){
        println(">=" + key  + "and <" + (key+1) + ":" + diffMap(key))
      } else {
        println(">=" + key  + "and <" + (key+1) + ":" + 0)
      }
    }

    if (diffMap.exists(_._1 == 4)){
      println(">=" + 4  + ":" + diffMap(4))
    } else {
      println(">=" + 4  + ":" + 0)
    }

    println("RMSE = " + Math.sqrt(MSE))

    println("The total execution time taken is "+ (System.nanoTime - t1) / 1e9d + " sec.")

  }
}