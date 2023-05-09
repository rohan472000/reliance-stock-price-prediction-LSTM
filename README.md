# Reliance Stock Price Prediction using LSTM Model

This Google Colab notebook contains an implementation of an LSTM model to predict the future stock prices of Reliance Industries Limited (RIL). The model is trained on historical data spanning from January 2018 to April 2023, and it generates forecasts for the next 10 trading days.

![Screenshot (186)](https://user-images.githubusercontent.com/96521078/236998369-f9b2e25e-27ff-4480-b33f-f3e400447c9a.png)

## Usage

- Open the Google Colab notebook in your web browser.
- Follow the instructions in the notebook to install the required packages, load the stock data, preprocess it, and train the LSTM model.
- Once the model is trained, you can use it to generate predictions for the next 10 trading days by running the appropriate code cells in the notebook.
- You can visualize the predicted stock prices and compare them with the actual prices by running the code cells that generate plots.
- You can modify the code and experiment with different hyperparameters to improve the accuracy of the predictions.

## Automated prediction and saving:

- To run `predict.py` use python3.8 (recommended).
- Run the `predict.py` to schedule the prediction function to run automatically every day. This function train the model, the preprocessed data, and the file path to save the predicted values as input, `action.yml` used here to schedule the prediction (`predict.py`) to run every day at a specific time.
- The predicted values will be saved in the specified Excel file. You can open this file to view the predicted values for each day. Note that the predicted values should be used for informational purposes only, and they should not be relied upon for making financial decisions.

**Note that the model's predictions should be used for informational purposes only, and they should not be relied upon for making financial decisions.**
