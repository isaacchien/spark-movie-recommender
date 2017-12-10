import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.log4j.{Level, Logger}

object Isaac_Chien_task2 {
  def main(args:Array[String]) {
    val t1 = System.nanoTime

    // Load and parse the data
    val conf = new SparkConf()
    conf.setMaster("local")
    conf.setAppName("CF")
    val sc = new SparkContext(conf)

    val testInput = sc.textFile(args(1))

    val userMovieTest = testInput
      .filter(line => line != "userId,movieId")
      .map(line => line.split(","))
      .map(word => ((word(0).toInt, word(1).toInt), word(1).toInt)) // ((userId, movieId)

    val inputData = sc.textFile(args(0))
    val data = inputData
      .filter(line => line != "userId,movieId,rating,timestamp")
      .map(line => line.split(","))

    val userMovieRating = data
      .map(word => ((word(0).toInt, word(1).toInt), word(2).toDouble)) // ((userId, movieId), rating)
      .subtractByKey(userMovieTest)

    val userMoviePredict = userMovieTest
      .map(word => (word._1._1, word._1._2)) // ((userId, movieId)
      .collect()

    val userMovieRatingMap = userMovieRating.collectAsMap()

    val userMovies = userMovieRating // user, movies, average
      .map(word => ((word._1._1), Set(word._1._2)))
      .reduceByKey((a, b) => (a ++ b))
      .map(userMovie => {
        val userA = userMovie._1
        val moviesUserA = userMovie._2
        val sumUserA = moviesUserA
          .toSeq
          .map(userMovieRatingMap(userA, _))
          .sum
        val ra = sumUserA / moviesUserA.size
        (userA,(userMovie._2, ra))
      })


    val userMoviesMap = userMovies.collectAsMap()


    val usersWuv = userMovies // (userA, userU), wuv
      .cartesian(userMovies)
      .filter{
        case (a,b) => a._1 < b._1
      }
      .map(pair => {
        val sharedMovies = pair._1._2._1.intersect(pair._2._2._1)

        val userU = pair._1._1
        val userV = pair._2._1

        val sumU = sharedMovies
          .map(userMovieRatingMap(userU, _))
          .sum

        val sumV = sharedMovies
          .toList
          .map(userMovieRatingMap(userV, _))
          .sum

        val ru = sumU / sharedMovies.size
        val rv = sumV / sharedMovies.size


        val numerator = sharedMovies
          .map( movieId => {
            (userMovieRatingMap(userU, movieId) - ru) * (userMovieRatingMap(userV, movieId) - rv)
          } )
          .sum


//        ((userU, userV),numerator)
        val denomLeft = sharedMovies
          .map( movieId => {
            Math.pow(userMovieRatingMap(userU, movieId) - ru, 2)
          } )
          .sum

        val denomRight = sharedMovies
          .map( movieId => {
            Math.pow(userMovieRatingMap(userV, movieId) - rv, 2)
          } )
          .sum

        val wuv = numerator / (Math.sqrt(denomLeft) * Math.sqrt(denomRight))

        (Set(pair._1._1, pair._2._1), if(wuv.isNaN) 0 else wuv) // pearson coorelation w
      })
//      .collect()
//      .foreach(println)


    println("predicting")
    val prediction = userMoviePredict
      .foreach(userMovie => {
        val userA = userMovie._1
        val movieI = userMovie._2

        val moviesUserA = userMoviesMap(userA)._1
        val ra = userMoviesMap(userA)._2

        // filter map to have only users that have rated moviei
        val otherUsers = userMovies // usermovies average
          .filter(userMovie => (userMovie._1 != userA))
          .filter(userMovie => (userMovie._2._1.contains(movieI)))


        val ruiRu = otherUsers // list of ruiru
          .map(userMovie => {
          val userU = userMovie._1
          val moviesUserU = userMovie._2._1
          println("userU: " + userU)

          val coratedMovies = moviesUserU.intersect(moviesUserA)

          val sum = coratedMovies.toSeq.map(userMovieRatingMap(userU, _)).sum
          val ru = (sum / userMovie._2._1.size)

          //get rating for that user
          val rui = userMovieRatingMap(userU, movieI)
          (Set(userA, userU), (rui - ru))
        })
        // TOO SLOW. MUST JOIN OR SOMETHING ELSE.
        println("joining ruiRuWuv")
        val ruiRuWuv = ruiRu
          .join(usersWuv)
//          .collect()
//          .foreach(println)
        println("ruiRuWuv size: " + ruiRuWuv.countApprox(2000, .5))
        println("numerator")
//        println("ra: " + ra)
        val numerator = ruiRuWuv.map(joined => {
            (joined._2._1 * joined._2._2)
          })
          .sum()

        println("numerator: " + numerator)

        val denominator = ruiRuWuv
          .map(joined => {
            Math.abs(joined._2._2)
          })
          .sum
        println("denominator: " + denominator)

        println("Predict: " + (userA, movieI), (numerator / denominator) + ra )

      })

  }
}