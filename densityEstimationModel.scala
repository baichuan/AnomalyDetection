// Author: Baichuan Zhang
// Mentor: Yanbo Liang
// Code was written during the 2016 summer intern at Spark team in Hortonworks

import org.apache.spark.ml.feature.{VectorAssembler, PCA}
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import scala.math._

// Model overview:

// Density Estimation Model: In this model, the idea is to evaluate per user, a probability density function from the observed sample data points.
// Each data point consists of dimensions equal to the number of features being selected.
// Before applying the algorithm, we normalize the training set by first computing the mean and standard deviation of each feature of the dataset and then subtracting actual data points from the mean and dividing the result by standard deviation.
// In our probability density estimation, we use Gaussian distribution function as the method for computing probability density.
// We assume that in reduced feature space, each feature is conditionally independent of one another, thus the final Gaussian probability density can be computed by factorizing each of the probability densities.

// -----------------------------------------------------------------------------

// Tuning parameters in the ML pipeline:
// 1) Reduced dimensionality in PCA
// 2) User-defined probability threshold in the density estimation model

object densityEstimationModel {

  def parameterEstimator(data: DataFrame, featureCol: String): Double = {

    val Array(trainData, testData) = data.randomSplit(Array(0.6, 0.4), 10L)

    val features = trainData.select(featureCol).rdd.map { case Row(v: Vector) => v }
    val featureSummary = features.aggregate(new MultivariateOnlineSummarizer())(
      (summary, feat) => summary.add(feat),
      (sum1, sum2) => sum1.merge(sum2))

    val meanVector = featureSummary.mean
    val varianceVector = featureSummary.variance

    val probThreshold = testData.select(featureCol).rdd.map { r =>
      val pcaFeatures = r.getAs[Vector](0)
      var t = 1.0
      (0 until pcaFeatures.size).foreach { i =>
        // compute a one-dimensional Gaussian function
        t = t * (1 / sqrt(2 * Pi * varianceVector(i))) * exp(-pow(pcaFeatures(i) - meanVector(i), 2) / (2 * varianceVector(i)))
      }
      t
    }.mean()
    probThreshold
  }

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("Density Estimation Model").setMaster("local[2]")
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

    val allDataRDD = dataVA1.select(VA1.getOutputCol).rdd.map { case Row(v: Vector) => v }
    val featureSummary1 = allDataRDD.aggregate(new MultivariateOnlineSummarizer())(
      (summary, feature) => summary.add(feature),
      (summary1, summary2) => summary1.merge(summary2)
    )
    val variances = featureSummary1.variance

    // perform feature selection to remove all the features with zero variance
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
    val Array(trainData, testData) = pcaData.randomSplit(Array(0.7, 0.3), 10L)

    // compute the mean and variance of each selected feature
    val selectedFeatures = trainData.select(pca.getOutputCol).rdd.map { case Row(v: Vector) => v }
    val featureSummary2 = selectedFeatures.aggregate(new MultivariateOnlineSummarizer())(
      (summary, feat) => summary.add(feat),
      (sum1, sum2) => sum1.merge(sum2))

    val meanVector = featureSummary2.mean
    val varianceVector = featureSummary2.variance

    val probThreshold = parameterEstimator(trainData, pca.getOutputCol)

    // Implement density estimation based anomaly detection model
    // 0.0 represents normal points and 1.0 represents predicted anomaly points
    val predictionLabel = testData.select(pcaModel.getOutputCol).rdd.map { r =>
      val pcaFeatures = r.getAs[Vector](0)
      var t = 1.0
      (0 until pcaFeatures.size).foreach { i =>
        // compute a one-dimensional Gaussian probability density function (pdf)
        t = t * (1 / sqrt(2 * Pi * varianceVector(i))) * exp(-pow(pcaFeatures(i) - meanVector(i), 2) / (2 * varianceVector(i)))
      }
      t
    }.map { x =>
      val prediction = if (x > probThreshold) 0.0 else 1.0
      prediction
    }

  }
}
