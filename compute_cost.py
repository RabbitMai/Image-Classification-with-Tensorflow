def compute_cost(Z3, Y):

    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Z3,labels=Y))

    return cost
