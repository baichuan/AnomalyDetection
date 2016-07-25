// Author: Baichuan Zhang
// Code was written during the 2016 summer intern at Spark team in Hortonworks

import org.apache.spark.ml.feature.{PCA, VectorAssembler}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import scala.math._

// Model overview:
// We assume that in reduced feature space, the anomaly points are far away any clusters from K-means clustering
// We utilize the Euclidean distance to compute the similarity between each test point and cluster centroid

// ------------------------------------------------------------------------------------------------

// Tuning parameters in the ML pipeline:
// 1) Reduced dimensionality in PCA
// 2) Number of clusters in k-means
// 3) User-defined distance threshold parameters in the ML pipeline:

object distanceKmeansModel {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("K-means Distance Model").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    // Load the data from file
    val rawData = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "false")
      .option("inferSchema", "true")
      .load("/Users/baichuan.zhang/project/AD/data/kaggle_test/parsedData.csv")

    // Use vector assembler to group all the features
    val columnNames = rawData.columns
    val VA1 = new VectorAssembler()
      .setInputCols(columnNames.slice(1, columnNames.length))
      .setOutputCol("features1")
    val dataVA1 = VA1.transform(rawData)

    // perform feature selection to remove all the features with zero variance
    val allDataRDD = dataVA1.select(VA1.getOutputCol).rdd.map {case Row(v: Vector) => v}
    val featureSummary = allDataRDD.aggregate(new MultivariateOnlineSummarizer())(
      (summary, feature) => summary.add(feature),
      (summary1, summary2) => summary1.merge(summary2)
    )
    val variances = featureSummary.variance

    val indices = scala.collection.mutable.MutableList[Int]()
    for (i <- 0 until variances.size) {
      if (variances(i) != 0.0){
        indices += i
      }
    }
    val arrayBuilder = scala.collection.mutable.ArrayBuilder.make[String]
    indices.foreach { i => arrayBuilder += columnNames(i+1) }
    val filteredColumnNames = arrayBuilder.result()

    // project the selected features into the data frame
    val VA2 = new VectorAssembler()
      .setInputCols(filteredColumnNames)
      .setOutputCol("features2")
    val dataVA2 = VA2.transform(dataVA1)

    // perform feature scaling for each selected feature
    val scaler = new StandardScaler()
      .setInputCol(VA2.getOutputCol)
      .setOutputCol("features2Norm")
      .setWithStd(true)
      .setWithMean(false)

    val scalerModel = scaler.fit(dataVA2)
    val dataVA3 = scalerModel.transform(dataVA2)

    // After we remove these low-variance columns, we apply Principal Component Analysis (PCA) for feature reduction.
    // In our case, some of the features for data points are highly correlated.
    // By applying PCA, we can find a linear combination of correlated features.
    // PCA also chooses the best combination from a set of possible linear functions that retains maximum variance of our data.

    val pca = new PCA()
      .setInputCol(scaler.getOutputCol)
      .setOutputCol("pcaFeatures")
      .setK(5)

    val pcaModel = pca.fit(dataVA3)
    val pcaData = pcaModel.transform(dataVA3)

    // split the reduced data as 70% training and 30% test
    // training is for model learning and test is for model evaluation
    val Array(trainData, testData) = pcaData.randomSplit(Array(0.7, 0.3), 10)

    // Get the optimal number of clusters based on SSE cost
    val optimalK = numOfClustersParameterEstimator(trainData, pca.getOutputCol)

    // run K-means model on the training set
    val kmeans = new KMeans()
      .setK(optimalK)
      .setFeaturesCol(pcaModel.getOutputCol)
      .setPredictionCol("prediction")

    val kmeansModel = kmeans.fit(trainData)
    val centroidVector = kmeansModel.clusterCenters

    // Compute the euclidean distance between each testData and each of cluster centroids
    // Select the min-distance in order to compare with a user-defined distance threshold to decide the anomaly points
    val distThreshold = distThresholdParameterEstimator(trainData, pca.getOutputCol)

    // 0.0 represents normal points and 1.0 represents predicted anomaly points
    val predictionLabel = testData.select(pcaModel.getOutputCol).rdd.map { r =>
      val testPoint = r.getAs[Vector](0).toArray

      var distMin = Double.MaxValue
      (0 until centroidVector.size).foreach { i =>
        val d = distance(testPoint, centroidVector(i).toArray)
        distMin = if (d < distMin) d else distMin
      }
      distMin
    }.map { x =>
      val pred = if (x  > distThreshold) 1.0 else 0.0
      pred
    }
  }

  // compute the Euclidean distance between two points
  def distance(a: Array[Double], b: Array[Double]): Double = {
    sqrt(a.zip(b).map(p => p._1 - p._2).map(d => d * d).sum)
  }

  // Estimate user-defined distance threshold using further partitioned training set
  def distThresholdParameterEstimator(data: DataFrame, featureCol: String): Double = {

    val Array(trainData, testData) = data.randomSplit(Array(0.6, 0.4), 10L)

    val kmeans = new KMeans()
      .setK(10)
      .setFeaturesCol(featureCol)
      .setPredictionCol("prediction")

    val kmeansModel = kmeans.fit(trainData)
    val centroidVector = kmeansModel.clusterCenters

    val threshold = testData.select(featureCol).rdd.map { r =>
      val dataPoint = r.getAs[Vector](0).toArray

      var distMin = Double.MaxValue
      (0 until centroidVector.size).foreach { i =>
        val dist = distance(dataPoint, centroidVector(i).toArray)
        distMin = if (dist < distMin) dist else distMin
      }
      distMin
    }.mean()

    threshold
    }

  // Estimate the number of cluster parameter in K-means using training set with smallest SSE (Sum Square Error)
  def numOfClustersParameterEstimator(data: DataFrame, featureCol: String): Int = {

    val numOfClustersSSE = (5 to 20 by 5).map { k =>
      val kmeans = new KMeans()
        .setK(k)
        .setFeaturesCol(featureCol)
        .setPredictionCol("prediction")
      val kmeansModel = kmeans.fit(data)
      val sseCost = kmeansModel.computeCost(data)
      (k, sseCost)
    }
    // return the k value with corresponding smallest SSE
    val optimalK = numOfClustersSSE.sortBy(_._2).map(_._1).toArray.apply(0)
    optimalK
  }

}