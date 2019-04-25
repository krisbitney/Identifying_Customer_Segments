# Identifying_Customer_Segments
An educational deep learning project. The goal is to identify customer segments by clustering customers in a reduced feature space. After producing a cluster model, I compared the population of customers to the general German population to determine how segments of the general population are represented in the customer population.

For readability, I split the project into two Jupyter notebooks.

### Step 1: Explore and preprocess the data

In the "Identify_Customer_Segments_Preprocessing.ipynb" Jupyter notebook, I complete the following tasks:
1. Explored data
2. Assessed missing values
    * Converted missing values to NaN
    * Assessed missing values by dataframe column and dropped outliers
    * Assessed missing values by dataframe row and dropped outliers
3. Selected, re-encoded, and engineered features
4. Created a data preprocessing pipeline

### Step 2: Reduce feature space dimensionality, find clusters, and identify customer segments

In the "Identify_Customer_Segments_Clustering.ipynb" Jupyter notebook, I complete the following tasks:
1. Imputed missing values using mean imputation (multiple imputation coming soon)
2. Standardized features
3. Reduced dimensionaly of feature space
    * Transformed feature space using principal components analysis (PCA)
    * Selected "best" subset of transformed features
    * Interpreted principal components using principal directions in original feature space
4. Clustered customer data using KMeans
5. Assigned general population to clusters and explored results

Principal components were selected based on variance explained and a hypothetical measure of the curse of dimensionality effect. Suppose our data were uniformly distributed within a unit hypersphere centered at the origin. With a given sample size and number of input features, we can calculate the median distance from the origin to a point (Hastie, Tibshirani, & Friedman, 2009).

Cluster model performance was evaluated using three metrics: mean square error, silhouette coefficient, and Calinski-Harabaz Score.
                  
 ### Required libraries                       
This project uses Numpy, Pandas, Pyplot, Seaborn, and Sklearn.
