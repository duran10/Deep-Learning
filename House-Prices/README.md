# House Prices Prediction with DNN Regressor
This project uses a deep neural network (DNN) regressor to predict house prices from data provided by Kaggle's House Prices: Advanced Regression Techniques competition. The goal is to build a model that achieves the lowest possible root mean squared error (RMSE) on the test set.

## Requirements
To run the Jupyter notebook house-prices-dnn-regressor.ipynb, you'll need:

- Python 3.6+
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow 2.x
- BayesianOptimization
- 
## Usage
To reproduce the results in the notebook, follow these steps:

1. Download the dataset from the competition website and extract it into the data folder.

2. Open the house-prices-dnn-regressor.ipynb notebook in Jupyter.

3. Run the notebook from top to bottom, executing each code cell sequentially.

4. The notebook will train a DNN regressor model on the training set using Bayesian optimization for hyperparameter tuning. It will perform cross-validation and evaluate the model on the test set, and generate a submission file in the correct format for the Kaggle competition.

5. You can submit the generated CSV file to Kaggle to get a score on the test set and compare it to the RMSE achieved in the notebook.

## Results
The DNN regressor model achieved a RMSE of 0.14 on the test set, which is a very competitive score in the Kaggle competition leaderboard. Bayesian optimization was used to tune the hyperparameters of the model, resulting in improved performance compared to the initial set of hyperparameters.

The model architecture consists of several dense layers with dropout regularization and batch normalization, followed by an output layer with linear activation. The model was trained using the Adam optimizer and mean squared error loss function.
