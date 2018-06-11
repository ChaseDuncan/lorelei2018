import tensorflow as tf
import numpy as np
import os

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float64_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def decode(serialized_example):
    """Parses an image and label from the given `serialized_example`."""
    features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'single_shp': tf.FixedLenFeature([3], dtype=tf.int64),
                'single': tf.VarLenFeature(tf.float32),
                'pair_shp': tf.FixedLenFeature([5], dtype=tf.int64),
                'pair':   tf.VarLenFeature(tf.float32),
                'truth':  tf.VarLenFeature(tf.int64)
                }
            )
    #single = features['single']
    single = tf.reshape(features['single'].values, features['single_shp'])
    #pair =   features['pair']
    # tf.reshape(tf.convert_to_tensor_or_sparse_tensor(
    pair = tf.reshape(features['pair'].values, features['pair_shp'])
    truth = features['truth'].values
    return single, pair, truth

def convert_to(npz_names, tfrecord_name):
    """Converts a dataset to tfrecords."""
    single_raw = []
    single_shapes = []
    pair_raw = []
    pair_shapes = []
    truth = []

    for fname in npz_names:
        with np.load(fname) as data:
            single = data['single']
            single_raw.append(np.ravel(single))
            single_shapes.append(single.shape)

            pair = data['pair']
            pair_raw.append(np.ravel(pair))
            pair_shapes.append(pair.shape)
            
            truth.append(data['truth'])

    print('Writing', tfrecord_name)

    with tf.python_io.TFRecordWriter(tfrecord_name) as writer:
        for s_shp, s, p_shp, p, t in zip(single_shapes, single_raw, pair_shapes, pair_raw, truth):
            example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'single_shp': _int64_feature(list(s_shp)),
                            'single': _float64_feature(s),
                            'pair_shp': _int64_feature(list(p_shp)),
                            'pair':   _float64_feature(p),
                            'truth':  _int64_feature(t)
                            })
                        )
            writer.write(example.SerializeToString())
   
if __name__=="__main__":
    import sys
    #path1 = '../data/tac2009/npz_files/'
    #path2 = '../data/tac2010/npz_files/'
    #npz_names = [path1 + fname for fname in os.listdir(path1)] + [path2 + fname for fname in os.listdir(path2)]
    #path = '../data/tac2017/npz_files/'
    #path = 'data/npz/'
    path = sys.argv[1]
    tfr_file = sys.argv[2] 
    npz_names = [path + fname for fname in os.listdir(path)]
    #convert_to(npz_names, 'data/tfr/2009_training.tfrecord')
    convert_to(npz_names, tfr_file)
