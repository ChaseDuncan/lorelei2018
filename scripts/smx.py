# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 12:53:09 2018

@author: manchan2
"""
import tensorflow as tf

def sort(z):
    return tf.contrib.framework.sort(z, direction='DESCENDING')

def largest_true_index(mask):
    """Find the largest true index in mask (based on 1-indexing)
    
    :param mask: boolean tensor of shape (n,)
    :return: tf.int32
    """
    # use tf.int32 because it is required for slicing
    indices = tf.range(1, tf.size(mask)+1, dtype=tf.int32)
    indices_true = tf.boolean_mask(indices, mask)
    # tensorflow reduce_max returns negative number if array is empty
    # correct to 0
    return tf.maximum(tf.reduce_max(indices_true), 0)

def smx(z, k, beta):
    """Take the soft k-max of z
    
    :param z: tf.float32 tensor of shape of (n,)
    :param k: natural number, 1 < k < n
    :param beta: regularization parameter, decreasing encourages spreading of attention
    """
    tf_beta = tf.constant(beta, dtype=tf.float32)
    tf_k = tf.constant(k, dtype=tf.float32)
    # First sort z
    sort_z = sort(z)
    return tf.reduce_sum(sort_z[:k])
    # Calculate R
    #r_array = tf.range(1, k, dtype=tf.float32)
    #exp_b_z = tf.exp(tf.scalar_mul(tf_beta, sort_z))
    #cumsum = tf.cumsum(exp_b_z, reverse=True)
    #numerator = cumsum[2:(k+1)]
    #print(numerator.get_shape)
    #rhs = tf.scalar_mul(tf.reciprocal(tf_beta), tf.log(numerator/(tf_k-r_array)))
    #lhs = sort_z[:(k-1)]
    #mask = tf.greater_equal(lhs, rhs)
    #R = largest_true_index(mask)
    ## Assign weight 1 to for z before R
    ## Log(mean(exp(z))) for z after R
    #k_minus_r = tf_k - tf.cast(R, tf.float32)
    #n_minus_r = tf.cast(tf.size(z), tf.float32) - tf.cast(R, tf.float32)
    #avg = tf.scalar_mul(k_minus_r/tf_beta,
    #              tf.log(tf.reduce_sum(exp_b_z[R:])/n_minus_r))
    #rest = tf.reduce_sum(sort_z[:R])
    #return avg + rest

#test smx function
if __name__=="__main__":
    z = tf.Variable([3, 4, 6, 5, 4, 4, 4, 4], dtype=tf.float32)
    sortz = sort(z)
    result = smx(z, 1, 10)
    grads = tf.gradients(result, z, stop_gradients=z)
    init = tf.global_variables_initializer()
    
    with tf.Session() as session:
        session.run(init)
        print(grads[0].eval(), result.eval())
