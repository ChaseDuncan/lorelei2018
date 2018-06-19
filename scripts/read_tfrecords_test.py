import tensorflow as tf
from write_tfrecords import decode
from train import StarModel
import matplotlib.pyplot as plt
from time import time

tf.logging.set_verbosity(tf.logging.INFO)

beta = 5
k = 6

def data_iterator(tfr_file, epoch=1):
    """ """
    dataset = tf.data.TFRecordDataset(tfr_file)
    dataset = dataset.map(decode)
    # TODO: does this value make sense for buffer size?
    dataset = dataset.shuffle(1000)
    return dataset.make_initializable_iterator()

if __name__=="__main__":
    import sys
    k = int(sys.argv[1])
    epoch = 50
    star = StarModel(4, 11, k, beta)

    test_it = data_iterator('/home/cddunca2/lorelei2018/data/tfr/nitish_2017.tfrecord')
    s,p,t = test_it.get_next()
    accuracy = star.accuracy(s,p,t)

    training_it = data_iterator('/home/cddunca2/lorelei2018/data/tfr/nitish_2016.tfrecord',epoch=epoch)
    single, pair, truth = training_it.get_next()
    loss = star.loss(single, pair, truth)
    optimizer = tf.train.AdagradOptimizer(0.01)
    minimizer = optimizer.minimize(loss)

    #initialize_iter = initializable_iterator.initializer
    
    init_op = tf.global_variables_initializer()
    epoch_loss = []
    dataset_size = 56
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        start_time = time()
        training_loss = []

        test_err = []
        for step in range(epoch):
            sess.run(training_it.initializer)
            epoch_loss = []
            while True:
                try:
                    _, loss_val = sess.run([minimizer, loss])
                    epoch_loss.append(loss_val)
                except tf.errors.OutOfRangeError:
                    print('done', step)
                    break

            training_loss.append(epoch_loss)

            epoch_test_err = []

            sess.run(test_it.initializer)
            while True:
                try:
                    epoch_test_err.append(sess.run(accuracy))
                except tf.errors.OutOfRangeError:
                    break  
            test_err.append(sum(epoch_test_err)/float(len(epoch_test_err)))
        model_path = "models/globerson_model_k"+str(k)+"_hard"    
        saver.save(sess, model_path)
        print('train_time', time()-start_time)
        dataset_size = len(epoch_loss)
        average_of_per_epoch_training_loss = \
            [sum(i)/dataset_size for i in training_loss]
        #plt.plot(average_of_per_epoch_training_loss, label="train_err")
        print(average_of_per_epoch_training_loss)
        #plt.show()
        #plt.plot(test_err,label="test_err")
        #plt.show()
        #print(test_err)


