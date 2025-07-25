# Real-Time Twitter Sentiment Analysis with PySpark and Kafka

![Project Status](https://img.shields.io/badge/status-completed-green)
![Tech Stack](https://img.shields.io/badge/tech-PySpark%2C%20Kafka%2C%20Docker-blue)

This project implements a complete end-to-end pipeline for performing sentiment analysis on Twitter data. It consists of two main parts:

1.  **Batch Model Training:** A PySpark ML pipeline to train and evaluate multiple classification models on the Sentiment140 dataset (1.6 million tweets).
2.  **Real-Time Inference:** A streaming application that deploys the trained model to perform live sentiment analysis on tweets ingested through Kafka.

## System Architecture

The core of the project is the real-time streaming pipeline, which leverages Spark Structured Streaming to apply the pre-trained sentiment model to live data.

<p align="center">
  <img src="https" width="800" alt="System Architecture Diagram">
</p>

---

## Part 1: Model Training & Evaluation

This repository contains the notebooks and scripts for the batch training phase.

### Methodology

We systematically evaluated several classification models by testing them against various feature engineering techniques to find the optimal combination for sentiment prediction.

*   **Models Evaluated:**
    *   Logistic Regression
    *   Support Vector Machines (Linear Kernel)
    *   Naive Bayes
    *   Random Forest & Decision Tree
*   **Feature Engineering Techniques:**
    *   Hashing TF-IDF vs. CountVectorizer + TF-IDF
    *   N-grams (1-Gram, 2-Gram, 3-Gram)
    *   Feature Selection with ChiSqSelector

### Key Results

The top-performing model was **Logistic Regression** using a combination of **CountVectorizer, TF-IDF, n-grams (1-3), and ChiSqSelector**, achieving an **F1-Score of 0.808**.

<p align="center">
  <b>Table 1: Model Accuracy Across Different Feature Sets</b><br>
  <img src="" width="600"/>
</p>
<br>
<p align="center">
  <b>Table 2: Detailed Performance Metrics for Final Models</b><br>
  <img src="" width="600"/>
</p>

### Running at Scale on Google Cloud

This project is configured to run on a cloud platform. The `cluster_logistic_job.py` script is provided to demonstrate how to submit the training job to a **Google Cloud Dataproc** cluster for distributed processing.

---

## Part 2: ETL Pipeline & Live Sentiment Analysis

The real-time inference pipeline is a separate, containerized application. It uses the trained Logistic Regression model to classify tweets from a live Kafka stream and stores the results in a Delta Lake.

**[>> View the Real-Time ETL Repository Here <<](https://github.com/Wazzabeee/pyspark-etl-twitter/tree/main)**