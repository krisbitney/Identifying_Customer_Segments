# Identifying_Customer_Segments
An educational deep learning project. The goal is to identify customer segments by clustering customers in a reduced feature space. After producing a cluster model, I compared the population of customers to the general German population to determine how segments of the general population are represented in the customer population.

For readability, I split the project into two Jupyter notebooks.

### Step 1: Explore and preprocess the data. 
##### See "Identify_Customer_Segments_Preprocessing.ipynb". 

I complete the following tasks:
1. Explore data
2. Assess missing values
    * Convert missing values to NaN
    * Assess missing values by dataframe column and drop outliers
    * Assess missing values by dataframe row and drop outliers
3. Select, re-encode, and engineer features
4. Create a data preprocessing pipeline

### Step 2: Reduce feature space dimensionality, find clusters, and identify customer segments. 
##### See "Identify_Customer_Segments_Clustering.ipynb". 

I complete the following tasks:
1. Impute missing values using mean imputation (multiple imputation coming soon)
2. Standardize features
3. Reduce dimensionaly of feature space
    * Transform feature space using principal components analysis (PCA)
    * Select "best" subset of transformed features
    * Interpret principal components using principal directions in original feature space
4. Cluster customer data using KMeans
5. Assign general population to clusters and explore results

Principal components were selected based on variance explained and a hypothetical measure of the curse of dimensionality effect. Suppose our data were uniformly distributed within a unit hypersphere centered at the origin. With a given sample size and number of input features, we can calculate the median distance from the origin to a point (Hastie, Tibshirani, & Friedman, 2009).

Cluster model performance was evaluated using three metrics: mean square error, silhouette coefficient, and Calinski-Harabaz Score.
                  
 ### Required libraries                       
This project uses Numpy, Pandas, Pyplot, Seaborn, and Sklearn.
