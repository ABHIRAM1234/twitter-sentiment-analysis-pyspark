# Twitter Sentiment Analysis (PySpark)
## About
This repo contains all the notebooks used for sentimental analysis on the [Sentiment140](http://help.sentiment140.com/for-students) dataset.
It was developed part of an end-of-term project for 8INF919 : Machine Learning for Big Data at [UQAC](https://www.uqac.ca/) in collaboration with [Thomas Sirvent](https://github.com/LargeWaffle).

You can find in the repo the LaTeX report and the presentation slides associated to this project (in french). If you'd like to read english explanations check it out my [website](https://clementdelteil.com/projects/1-twitter-sentiment). 

## Models used
We worked with the following models :
- Logistic Regression
- Support Vector Machines (Linear Kernel)
- Naive Bayes
- Random Forest
- Decision Tree
- Random Forest
 
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
     
## ETL Pipeline & Live Sentiment Analysis
Another part of this project was to implement an [ETL Pipeline with Live Sentiment Analysis](add link) using Spark Streaming / Apache Kafka and Docker. You can check it out here.

