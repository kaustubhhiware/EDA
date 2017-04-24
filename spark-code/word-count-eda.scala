/* this code counts the frequency of words in a file (word-count.txt) */

/*this line is for loading data from disk and storing in the RDD (data). Note that textFile() reads one line at a time */
val data = sc.textFile("word-count.txt")

/* this line converts the lines into sequence of words. The expression \\s+ inside split breaks the lines into words delimited by one or more space */
val words = data.flatMap(line => {line.split("\\s+")})

/* again we convert words only RDD to (word,1) pairs suitable for reduceByKey function */
val paired = words.map(word => (word, 1))

/*we now actually count the words by reducing it basd on key (word) */
val counts = paired.reduceByKey{(x, y) => x + y}

/* finally save the result into a file output.txt */
counts.saveAsTextFile("output.txt")
