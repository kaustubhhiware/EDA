/* this code computes the k-nearest vector of a query vector from a number of vectors.  
the input data can be found in knn.txt. one line for one vector. the first field is the 
vector id and rest fields are the values. */

/* read the data */
val data = sc.textFile("knn.txt")

/* dimension of vector. i set it to 3. it depends on your data */
val dim = 3

/* here we are splitting a line, convert to double value and store in a array as vector, we also store id */
val vectors = data.map(line => {
	var temp = line.split("\\s+")
	var docid = temp(0)
	var elements = new Array[Double](dim)
	for(i <- 1 to temp.length-1) 
	{
	   elements(i-1) = temp(i).toDouble
	}
	(docid, elements)
     }
   )

/* array for query vector */
var qvec = new Array[Double](dim)

/* randomly create a query vector for which you compute the k-nearest neighbor */
val r = scala.util.Random
for(i <- 0 to dim-1) {
   qvec(i) = (1.0*r.nextInt(100))/100;
}

/* here we compute the cosine similarity between the query vector and the data vectors */
val score = vectors.map(t => {
	var sum = 0.0
	var length1 = 0.0
	var length2 = 0.0
	var dim = qvec.length
	for(i <- 0 to dim-1) {
		sum += qvec(i) * t._2(i)
		length1 += qvec(i) * qvec(i)
		length2 += t._2(i) * t._2(i)
	}
	
	var score = sum/(Math.sqrt(length1) * Math.sqrt(length2))
	(score, t._1)
    }
  )

/* take top 10 vector and print their, score and id */
score.top(10).foreach(println)
    
