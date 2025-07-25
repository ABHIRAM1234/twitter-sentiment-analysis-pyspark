# Twitter Sentiment Analysis (PySpark)
## About
This repo contains all the notebooks used for sentimental analysis on the [Sentiment140](http://help.sentiment140.com/for-students) dataset with PySpark.
It was developed as part of my course: Big Data

## Models used
We worked with the following models :
- Logistic Regression
- Support Vector Machines (Linear Kernel)
- Naive Bayes
- Random Forest
- Decision Tree
 
## Features tested
- Hashing TF-IDF
- Count Vectorizer TF-IDF
- ChisQSelector
- 1-Gram, 2-Gram, 3-Gram

## Results

<img
     src="https://github.com/Wazzabeee/twitter-sentiment-analysis/blob/main/images/features.png"
     />

<img
     src="https://github.com/Wazzabeee/twitter-sentiment-analysis/blob/main/images/summary.png"
     />
    
## Google Cloud Cluster (Dataproc)
In the notebooks directory, you'll find the a Python file called "cluster_logistic_job.py" if you are curious and you want to see how we ran our models in the Cloud. 

## ETL Pipeline & Live Sentiment Analysis
Another part of this project was to implement an ETL Pipeline with Live Sentiment Analysis using our pre-trained model, Spark Streaming, Apache Kafka and Docker. [The repository for this part is available here](https://github.com/Wazzabeee/pyspark-etl-twitter/tree/main).
