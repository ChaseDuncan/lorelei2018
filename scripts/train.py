from smx import smx
import tensorflow as tf
import numpy as np

class StarModel:
    def __init__(self, s, p, k, beta):
        self.single_weights = tf.Variable(np.ones(s), dtype=tf.float32,name="unary")
        self.pair_weights = tf.Variable(np.ones(p), dtype=tf.float32,name="pairwise")
        self.k = k
        self.beta = beta

    def loss(self, single_feats, pair_feats, truth):
        # multiply by weights
        single_multiplied = tf.einsum('nis,s->ni', single_feats, self.single_weights)
        pair_multiplied = tf.einsum('nmijs,s->nmij', pair_feats, self.pair_weights)
        
        # calculate q_{i,j} = max_j s_{i,j} + s{j}
        # single_multiplied has shape mxc
        # pair_multiplied has shape mx(m-1)xcxc
        # q will have shape mx(m-1)xc
        # where m is num_mentions and c is num_cands
        m = tf.shape(single_multiplied)[0]
        def slice_and_stack(j, m):
            def range_not_j(j, m):
                return tf.setdiff1d(tf.range(m), [j])
            return tf.gather(single_multiplied, range_not_j(j, m), axis=0)
        j_scores = tf.map_fn(lambda j: slice_and_stack(j, m), tf.range(m), dtype=tf.float32)
        # j scores has shape mx(m-1)x(m-1)xc, should have mx(m-1)xc. remove a dim
        # couldn't figure out why, but will simply remove the extra dim
        j_scores = j_scores[:,0,:,:]
        print(j_scores)
        q = tf.reduce_max(j_scores[:,:,tf.newaxis,:] + pair_multiplied, axis=3)
        # map the smx function over every row of qi
        # tf.stack(..., axis=1) is makes a matrix where each column is a vector in ...
        def fn(vec):
            return smx(vec, self.k, self.beta)
        smx_q = tf.map_fn(lambda x: tf.map_fn(fn, tf.transpose(x)), q)
        # #calculate the loss
        scores = single_multiplied + smx_q
        # ts = scores[tf.range(m),tf.cast(truth, tf.int32)]
        zip_range_truth = tf.stack([tf.range(m), tf.cast(truth, tf.int32)], axis=1)
        # true scores
        ts = tf.gather_nd(scores, zip_range_truth)
        # zero-one-loss
        def one_hot(t):
            return tf.one_hot(tf.cast(t, tf.int32), tf.shape(scores)[1], on_value = np.float32(0.), off_value = np.float32(1.))
        zero_one = tf.map_fn(lambda x: one_hot(x), truth, dtype=tf.float32)
        loss = tf.reduce_max(scores-ts[:,tf.newaxis]+zero_one, axis=1)
        return tf.reduce_sum(loss)

    def accuracy(self, single_feats, pair_feats, truth):
        # multiply by weights
        single_multiplied = tf.einsum('nis,s->ni', single_feats, self.single_weights)
        pair_multiplied = tf.einsum('nmijs,s->nmij', pair_feats, self.pair_weights)

        # calculate q_{i,j} = max_j s_{i,j} + s{j}
        # single_multiplied has shape mxc
        # pair_multiplied has shape mx(m-1)xcxc
        # q will have shape mx(m-1)xc
        # where m is num_mentions and c is num_cands
        m = tf.shape(single_multiplied)[0]
        def slice_and_stack(j, m):
            def range_not_j(j, m):
                return tf.setdiff1d(tf.range(m), [j])
            return tf.gather(single_multiplied, range_not_j(j, m), axis=0)
        j_scores = \
            tf.map_fn(lambda j: slice_and_stack(j, m), tf.range(m), dtype=tf.float32)
        # j scores has shape mx(m-1)x(m-1)xc, should have mx(m-1)xc. remove a dim
        # couldn't figure out why, but will simply remove the extra dim
        j_scores = j_scores[:,0,:,:]
        # print(j_scores)
        q = tf.reduce_max(j_scores[:,:,tf.newaxis,:] + pair_multiplied, axis=3)
        # map the smx function over every row of qi
        # tf.stack(..., axis=1) is makes a matrix where each column is a vector in ...
        def fn(vec):
            return smx(vec, self.k, self.beta)
        smx_q = tf.map_fn(lambda x: tf.map_fn(fn, tf.transpose(x)), q)
        # #calculate the loss
        scores = single_multiplied + smx_q
        argm = tf.argmax(scores, 1) # max for each column
        equals = tf.equal(argm, truth)
        return tf.reduce_mean(tf.cast(equals, tf.float32))

    def inference(self, single_feats, pair_feats):
        # multiply by weights
        single_multiplied = tf.einsum('nis,s->ni', single_feats, self.single_weights)
        pair_multiplied = tf.einsum('nmijs,s->nmij', pair_feats, self.pair_weights)

        # calculate q_{i,j} = max_j s_{i,j} + s{j}
        # single_multiplied has shape mxc
        # pair_multiplied has shape mx(m-1)xcxc
        # q will have shape mx(m-1)xc
        # where m is num_mentions and c is num_cands
        m = tf.shape(single_multiplied)[0]
        def slice_and_stack(j, m):
            def range_not_j(j, m):
                return tf.setdiff1d(tf.range(m), [j])[1]
            return tf.gather(single_multiplied, range_not_j(j, m), axis=0)
        j_scores = \
            tf.map_fn(lambda j: slice_and_stack(j, m), tf.range(m), dtype=tf.float32)

        q = tf.reduce_max(j_scores[:,:,tf.newaxis,:] + pair_multiplied, axis=3)

        # map the smx function over every row of qi
        # tf.stack(..., axis=1) is makes a matrix where each column is a vector in ...

        def fn(vec):
            return smx(vec, self.k, self.beta)
        smx_q = tf.map_fn(lambda x: tf.map_fn(fn, tf.transpose(x)), q)
        # #calculate the loss
        scores = single_multiplied + smx_q
        argm = tf.argmax(scores, 1) # max for each column
        return argm

