from sklearn import datasets, linear_model

from returns_data import read_goog_sp500_data

xData, yData = read_goog_sp500_data()

# Set up a linear model to represent this
googModel = linear_model.LinearRegression()
googModel.fit(xData.reshape(-1,1), yData.reshape(-1,1))

# Find the coefficient and intercept of this linear model
print googModel.coef_
print googModel.intercept_

################################################################################
#
# Custom linear regression model
#
################################################################################
import tensorflow as tf

# Declare list of features, we only have one real-valued feature
def model(features, labels, mode):

    # Build a linear model and predict values
    W = tf.get_variable("W", [1], dtype=tf.float64)
    b = tf.get_variable("b", [1], dtype=tf.float64)

    y = W * features['x'] + b

    # Loss sub-graph
    loss = tf.reduce_sum(tf.square(y - labels))

    # Training sub-graph
    global_step = tf.train.get_global_step()

    optimizer = tf.train.FtrlOptimizer(1)

    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))

    # ModelFnOps connects subgraphs we built to the appropriate functionality.
    return tf.contrib.learn.ModelFnOps(mode=mode, predictions=y, loss=loss, train_op=train)

estimator = tf.contrib.learn.Estimator(model_fn=model)

input_fn = tf.contrib.learn.io.numpy_input_fn({"x": xData}, yData, batch_size=len(xData), num_epochs=10000)

fit = estimator.fit(input_fn=input_fn, steps=10000)

for variable_name in fit.get_variable_names():
    print variable_name , " ---> " , fit.get_variable_value(variable_name)

# evaluate our model
print(estimator.evaluate(input_fn=input_fn, steps=10))















































