import bnn
print(bnn.available_params(bnn.NETWORK_LFC))

classifier = bnn.PynqBNN(network=bnn.NETWORK_LFC)


def regression(x):

# 後でファイル化処理記述

    y = classifier.inference("/home/xilinx/bnn/data/image.images-idx3-ubyte")

    return y

'''
# LFC-BNN Call
def regression(x):
    W = tf.Variable(tf.zeros([784, 10]), name="W")
    b = tf.Variable(tf.zeros([10]), name="b")
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    return y, [W, b]
'''

