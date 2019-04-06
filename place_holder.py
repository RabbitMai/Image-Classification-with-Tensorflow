def create_placeholders(n_H0, n_W0, n_C0, n_y):

    X = tf.placeholder('float', shape=(None, n_H0, n_W0, n_C0))
    Y = tf.placeholder('float', shape= (None, n_y))

    return X, Y
