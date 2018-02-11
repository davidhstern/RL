import tensorflow as tf

def feedforward(inputs, hidden_units, num_outputs):

  #  hidden = tf.layers.dense(inputs=inputs, units=hidden_units, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=inputs, units=num_outputs)

    return (tf.nn.softmax(logits), logits)
