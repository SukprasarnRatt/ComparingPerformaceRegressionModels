# Comparing Performance of Regression Models

This project focuses on the practical application of various regression models and compares their performance using the R-square metric. It aims to identify the most effective model for predicting student performance based on several influencing factors.

## Dataset

The "Student Performance" dataset is utilized to analyze factors affecting academic achievements. It comprises records for 1,000 students with the following details:

### Variables
- **Hours Studied**: Total study hours for each student.
- **Previous Scores**: Scores from previous tests.
- **Extracurricular Activities**: Participation in extracurricular activities (Yes or No).
- **Sleep Hours**: Average daily sleep hours.
- **Sample Question Papers Practiced**: Number of sample question papers practiced.

### Target Variable
- **Performance Index**: An overall performance measure of each student.

### Data Source
- [Student Performance Dataset on Kaggle](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression)

## Libraries
- Numpy
- Matplotlib
- Pandas
- Scikit-learn

## Steps in the Practice
1. **Preparing the Dataset**: Handle missing data using the SimpleImputer from the sklearn library.
2. **Splitting the Dataset**: Divide the dataset into a training set (80%) and a test set (20%).
3. **Training Models**: Train various regression models including multiple linear regression, polynomial linear regression, support vector regression, decision tree regression, and random forest regression using the sklearn library.
4. **Predicting Performance Index**: Use the trained models to predict the Performance Index of students using test set data.
5. **Comparing Regression Models**: Evaluate and compare the models using the R-square metric.

## Conclusion
After comparing all models using the R-square metric, the results are as follows:
- **Multiple Linear Regression**: R-square = 0.9880686410711422
- **Polynomial Linear Regression**: R-square = 0.9877782381996749
- **Support Vector Regression**: R-square = 0.9872457183028401
- **Decision Tree Regression**: R-square = 0.9741945758256975
- **Random Forest Regression**: R-square = 0.9838317633473252

The analysis concludes that the **Multiple Linear Regression** model is the best fit for this dataset, demonstrating the highest R-square value among the evaluated models.
