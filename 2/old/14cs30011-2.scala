/*The following file is for computing the Pearson correlation coefficient*/
// run with - spark-shell -i 14cs30011-2.scala

// pearson coefficient = 
//  n * sigma( X[i] * Y[i] ) - sigma(X[i]) * sigma(Y[i]) divided by
//  {n * sigma(X[i]^2) - sigma(X[i])^2 }^0.5 * {n * sigma(Y[i]^2) - sigma(Y[i])^2 }^0.5

// load data
val data = sc.textFile("cor.txt")

val X_Y = data.map{ line =>
		(  line.split("\\s+")(0).toDouble,
		   line.split("\\s+")(1).toDouble
		)
	}

// keep track of X[i]^2 and Y[i]^2
val squares = data.map{ line =>
		( Math.pow(line.split("\\s+")(0).toDouble,2), 
		  Math.pow(line.split("\\s+")(1).toDouble,2)
		)
	}

// keep track of X[i] * Y[i]
val product = data.map{ line =>
		( 
			line.split("\\s+")(0).toDouble*line.split("\\s+")(1).toDouble
		)
	}

// #pairs
var n = X_Y.count()	

// store sigma(X[i]) and sigma(Y[i])
val sigma = X_Y.reduce{ (a,b) =>
		(
			a._1 + b._1,
			a._2 + b._2
		)
	}

// store sigma(X[i] * Y[i])
val sigmaXY = product.reduce{ (a,b) =>
		(
			a + b
		)
	}

// store sigma(X[i] ^ 2) and sigma(Y[i] ^ 2)
val squaredsigma = squares.reduce{ (a,b) =>
		(
			a._1 + b._1 ,
			a._2 + b._2
		)  
	}

var num = n*sigmaXY - sigma._1*sigma._2
var denom = Math.sqrt(n*squaredsigma._1-Math.pow(sigma._1,2)) * Math.sqrt(n*squaredsigma._2-Math.pow(sigma._2,2))

var rho = num / denom
print("\nThe computed Pearson correlation coefficient is ")
print(rho)