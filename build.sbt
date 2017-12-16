name := "Isaac_Chien_hw3"

version := "1.0"

scalaVersion := "2.11.7"

libraryDependencies ++= {
  val sparkVer = "1.6.1"
  Seq(
    "org.apache.spark" %% "spark-core" % sparkVer,
    "org.apache.spark" %% "spark-mllib" % sparkVer
  )
}
