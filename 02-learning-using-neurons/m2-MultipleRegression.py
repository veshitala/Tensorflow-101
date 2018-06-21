import numpy as np
from sklearn import datasets, linear_model

from returns_data import read_xom_oil_nasdaq_data

nasdaqData, oilData, xomData = read_xom_oil_nasdaq_data()

combined = np.vstack((nasdaqData , oilData)).T

xomNasdaqOilModel = linear_model.LinearRegression()

xomNasdaqOilModel.fit(combined, xomData)
xomNasdaqOilModel.score(combined, xomData)

print xomNasdaqOilModel.coef_
print xomNasdaqOilModel.intercept_

################################################################################
#
# Multiple regression
#
################################################################################
import tensorflow as tf

# Model linear regression y = W1x1 + W2x2 + b
nasdaq_W = tf.Variable(tf.zeros([1, 1]), name="nasdaq_W")
oil_W = tf.Variable(tf.zeros([1, 1]), name="oil_W")

b = tf.Variable(tf.zeros([1]), name="b")

nasdaq_x = tf.placeholder(tf.float32, [None, 1], name="nasdaq_x")
oil_x = tf.placeholder(tf.float32, [None, 1], name="oil_x")

nasdaq_Wx = tf.matmul(nasdaq_x, nasdaq_W)
oil_Wx = tf.matmul(oil_x, oil_W)

y = nasdaq_Wx + oil_Wx + b

y_ = tf.placeholder(tf.float32, [None, 1])

# Add summary ops to collect data
nasdaq_W_hist = tf.summary.histogram("nasdaq_weights", nasdaq_W)
nasdaq_W_hist = tf.summary.histogram("oil_weights", oil_W)

b_hist = tf.summary.histogram("biases", b)
y_hist = tf.summary.histogram("y", y)

cost = tf.reduce_mean(tf.square(y_ - y))

cost_hist = tf.summary.histogram("cost", cost)


train_step_ftrl = tf.train.FtrlOptimizer(1).minimize(cost)

all_x_nasdaq = nasdaqData.reshape(-1, 1)
all_x_oil = oilData.reshape(-1, 1)
all_ys = xomData.reshape(-1, 1)

dataset_size = len(oilData)

def trainWithMultiplePointsPerEpoch(steps, train_step, batch_size):

  init = tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(init)

    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./linearregression_demo3', sess.graph)

    for i in range(steps):

      if dataset_size == batch_size:
        batch_start_idx = 0
      elif dataset_size < batch_size:
        raise ValueError("dataset_size: %d, must be greater than batch_size: %d" % (dataset_size, batch_size))
      else:
        batch_start_idx = (i * batch_size) % dataset_size

      batch_end_idx = batch_start_idx + batch_size

      batch_x_nasdaq = all_x_nasdaq[batch_start_idx : batch_end_idx]
      batch_x_oil = all_x_oil[batch_start_idx : batch_end_idx]
      batch_ys = all_ys[batch_start_idx : batch_end_idx]

      feed = { nasdaq_x: batch_x_nasdaq, oil_x: batch_x_oil, y_: batch_ys }

      sess.run(train_step_ftrl, feed_dict=feed)

      # Write out histogram summaries
      result = sess.run(merged_summary, feed_dict=feed)
      writer.add_summary(result, i)

      # Print result to screen for every 500 iterations
      if (i + 1) % 500 == 0:
        print("After %d iteration:" % i)
        print("W1: %s" % sess.run(nasdaq_W))
        print("W2: %s" % sess.run(oil_W))
        print("b: %f" % sess.run(b))

        print("cost: %f" % sess.run(cost, feed_dict=feed))
    writer.close()    

trainWithMultiplePointsPerEpoch(5000, train_step_ftrl, len(oilData))




































