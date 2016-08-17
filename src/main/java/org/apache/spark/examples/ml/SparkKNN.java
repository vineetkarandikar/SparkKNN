package org.apache.spark.examples.ml;

/*
 * This class takes the training data-set and predict result based upon test data-set.  
 * This program can run both on windows and linux.
 * @Author-Vineet Karandikar
 * 
 */

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.PairFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import scala.Tuple2;

public class SparkKNN {
	private static final Logger LOGGER = LoggerFactory.getLogger(SparkKNN.class);

	public static void main(String[] args) {
		SparkConf sparkConf = new SparkConf().setMaster("local").setAppName("KNN Spark");
		JavaSparkContext sc = new JavaSparkContext(sparkConf);
		System.setProperty("hadoop.home.dir", "D:/Softwares/winutils-master/winutils-master/hadoop-2.7.1");// only
																											// for
																											// windows.
		String trainData = "/home/ubuntu/imageFinaloutput.txt";
		String testData = "/home/ubuntu/imageToTest.txt";
		JavaPairRDD<String, String> trainDataMapPair1 = sc.textFile(trainData)
				.mapToPair(new PairFunction<String, String, String>() {

					private static final long serialVersionUID = 1L;

					public Tuple2<String, String> call(String arg0) throws Exception {
						String key = arg0.substring(0, 1);
						String value = arg0.substring(2);
						value = value.trim();
						value = value.replace(" ", ",");
						// System.out.println("\n key==> " + key + " value==>" +
						// value);
						return new Tuple2<String, String>(key, value);
					}
				});
		/// trainDataMapPair1.saveAsTextFile("D:/personal/image-recog/trainDataMapPair1.txt");
		JavaPairRDD<String, String> testDataMapPair1 = sc.textFile(testData)
				.mapToPair(new PairFunction<String, String, String>() {

					private static final long serialVersionUID = 1L;

					public Tuple2<String, String> call(String arg0) throws Exception {
						String value = arg0;
						value = value.replace(" ", ",");
						// System.out.println("\n key==> x" + " value==>" +
						// value);
						return new Tuple2<String, String>("x", value);
					}
				});
		// testDataMapPair1.saveAsTextFile("D:/personal/image-recog/testDataMapPair1.txt");
		JavaPairRDD<Tuple2<String, String>, Tuple2<String, String>> cartiseanResult = trainDataMapPair1
				.cartesian(testDataMapPair1);
		LOGGER.info(" " + cartiseanResult.count());
		// cartiseanResult.saveAsTextFile("D:/personal/image-recog/cartiseanResult.txt");
		JavaPairRDD<Double, String> resultStage1 = cartiseanResult
				.mapToPair(new PairFunction<Tuple2<Tuple2<String, String>, Tuple2<String, String>>, Double, String>() {
				
					private static final long serialVersionUID = 1L;

					public Tuple2<Double, String> call(Tuple2<Tuple2<String, String>, Tuple2<String, String>> arg0)
							throws Exception {
						String destinationKey = arg0._1._1;
						String sourceKey = arg0._2._1;
						String sourceValue = arg0._1._2;
						String destinationValue = arg0._2._2;
						String key = sourceKey + destinationKey;
						String sourceDataSplit[] = sourceValue.split(",");
						String destinationDataSplit[] = destinationValue.split(",");
						double sum = 0.0;
						for (int i = 0; i < sourceDataSplit.length; i++) {
							sum = sum + Math.pow(Double.parseDouble(sourceDataSplit[i])
									- Double.parseDouble(destinationDataSplit[i]), 2);
						}
						return new Tuple2<Double, String>(Math.sqrt(sum), key);
					}
				});
		List<Tuple2<Double, String>> resultStage2 = resultStage1.sortByKey(true).take(9);

		Map<String, Integer> labelCount = new HashMap<>();
		for (Tuple2<Double, String> k : resultStage2) {
			Integer n = labelCount.get(k._2());
			n = (n == null) ? 1 : ++n;
			labelCount.put(k._2(), n);
		}
		int maxValueInMap = (Collections.max(labelCount.values()));
		for (Entry<String, Integer> entry : labelCount.entrySet()) {
			if (entry.getValue() == maxValueInMap) {
				LOGGER.info("\n Predicted value ==> " + entry.getKey());
			}
		}
		sc.close();
	}

}
