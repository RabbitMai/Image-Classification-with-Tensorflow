def model(X_trian,Y_train,X_test,Y_test,learning_rate=0.009, num_epochs=100,\
          minibatch_size=256,print_cost=True):
    """
    Arguments:
    X_train -- training set, of shape (None, 64, 64, 3)
    Y_train -- test set, of shape (None, n_y = 2)
    X_test -- training set, of shape (None, 64, 64, 3)
    Y_test -- test set, of shape (None, n_y = 2)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (m,n_H0,n_W0,n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []

    #obtain inputs
    X,Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

    parameters = initialize_parameters()

    Z3 = forward_propagation(X, parameters)

    cost = compute_cost(Z3, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)
        
        for epoch in range(num_epochs):

            mini_cost = 0
            num_minibatches = int(m/minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, \
                                              minibatch_size, seed)

            for minibatch in minibatches:

                (X_minibatch, Y_minibatch) = minibatch
                _,temp_cost = sess.run([optimizer,cost],{X:X_minibatch,Y:Y_minibatch})

                mini_cost += temp_cost / num_minibatches

            #print cost in every 5 epoches
            if print_cost==True and epoch%5 == 0:
                print("Cost after epoch %i: %f" % (epoch, mini_cost))
            if print_cost==True and epoch%1 == 0:
                costs.append(mini_cost)

        #visualize the costs over epoches
        plt.plot(np.squeeze(costs))
        plt.ylabel('costs')
        plt.xlabel('iterations per(tens)')
        plt.title('learning rate: {}'.format(learning_rate))
        plt.show()

        #obtain predictions
        prediction_op = tf.argmax(Z3,1)
        correct_prediction = tf.equal(prediction_op,tf.argmax(Y,1))

        #calcualte accuracy for train and test sets
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
        print(accuracy)
        train_accuracy = accuracy.eval({X:X_train,Y:Y_train})
        test_accuracy = accuracy.eval({X:X_test,Y:Y_test})
        print("train accuracy: {}".format(train_accuracy))
        print("test accuracy: {}".format(test_accuracy))

        return train_accuracy, test_accuracy, parameters
