/*The following file is for computing the mean vector for each class */
// run with - spark-shell -i mean_vec.scala

// Author:  Kaustubh Hiware
// @kaustubhhiware

// load data
val data = sc.textFile("mean.txt")

val vectors = data.map{ line =>
		(line.split("\\s+")(0).toInt, 
			List( line.split("\\s+")(1).toDouble,
				  line.split("\\s+")(2).toDouble,
				  line.split("\\s+")(3).toDouble )
		)
	}

// count #features per class
val n = data.map{ line =>
		(line.split("\\s+")(0).toInt,1)
	}

val count = n.reduceByKey{
		(a,b) => a+b
	}.coalesce(1).sortByKey(true)

val mapCount = count.collectAsMap

// store the sum of all vectors here
val sum = vectors.reduceByKey{ (a, b) =>
		List(a(0) + b(0),a(1) + b(1),a(2) + b(2))
	}.coalesce(1).sortByKey(true)

val mean = sum.map{ x =>
		( x._1,
			List( x._2(0) / mapCount(x._1), x._2(1) / mapCount(x._1), x._2(2)/mapCount(x._1))
		) 
	}

// print all means
// convert to float to get 6th digit accuracy only
print("\nMean vectors for each class are as follows :\n")
print("Class: Mean_1        Mean_2        Mean_3\n")
for (i <- mean) {
	println(i._1+":    "+i._2(0).toFloat+"    "+i._2(1).toFloat+"    "+i._2(2).toFloat)
}

/* finally save the result into a file output.txt */
mean.saveAsTextFile("output-1.txt")