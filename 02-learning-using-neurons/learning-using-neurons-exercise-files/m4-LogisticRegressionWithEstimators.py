import pandas as pd 
import numpy as np
import statsmodels.api as sm
from returns_data import read_goog_sp500_logistic_data

xData, yData = read_goog_sp500_logistic_data()

logit = sm.Logit(yData, xData)

# Fit the Logistic model
result = logit.fit()

# All values >0.5 predict an up day for Google
predictions = (result.predict(xData) > 0.5)

# Count the number of times the actual up days match the predicted up days
num_accurate_predictions = (list(yData == predictions)).count(True)

pctAccuracy = float(num_accurate_predictions) / float(len(predictions))

print "Accuracy: ", pctAccuracy

################################################################################
#
# Logistic regression with estimators
#
################################################################################
import tensorflow as tf

features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

estimator = tf.contrib.learn.LinearClassifier(feature_columns=features)

# All returns in a 2D array
# [[-0.02184618]
# [ 0.00997998]
# [ 0.04329069]
# [ 0.03254923]
# [-0.01781632]]
x = np.expand_dims(xData[:,0], axis=1)

# True/False values for up/down days in a 2D array
# [[False]
# [ True]
# [ True]
# [ True]
# [ True]
# [False]]
y = np.expand_dims(np.array(yData), axis=1)

# Batch size of 100 and 10000 epochs
input_fn = tf.contrib.learn.io.numpy_input_fn({"x" : x}, y, batch_size=100, num_epochs=10000)

fit = estimator.fit(input_fn=input_fn, steps=10000)

# All data points in a single batch with just one epoch
input_fn_oneshot = tf.contrib.learn.io.numpy_input_fn({"x": x }, y, batch_size=len(x), num_epochs=1)

results = fit.evaluate(input_fn=input_fn_oneshot, steps=1)

print results

for variable_name in fit.get_variable_names():
    print variable_name , " ---> " , fit.get_variable_value(variable_name)




































