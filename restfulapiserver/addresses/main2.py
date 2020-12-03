
from liveness_dl_feeling import demo
from model import train_model, valid_model
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf
import argparse

ap = argparse.ArgumentParser()

ap.add_argument("-v", "--video", type=str, required=True, help="path to video path")

args = vars(ap.parse_args())


flags = tf.app.flags
flags.DEFINE_string('MODE', 'demo',
                    'Set program to run in different mode, include train, valid and demo.')
flags.DEFINE_string('checkpoint_dir', './ckpt',
                    'Path to model file.')
flags.DEFINE_string('train_data', './data/fer2013/fer2013.csv',
                    'Path to training data.')
flags.DEFINE_string('valid_data', './valid_sets/',
                    'Path to training data.')
flags.DEFINE_boolean('show_box', False,
                     'If true, the results will show detection box')

#  추가 ---------------------------------
flags.DEFINE_string('DETECTOR', 'dl/face_detection_model',
                    'path to OpenCVs deep learning face detector')
flags.DEFINE_string( 'MODEL', 'liveness.model', 'path to trained model')
flags.DEFINE_string( 'LE', 'le.pickle', 'path to label encoder')
flags.DEFINE_string( 'EMBEDDING_MODEL', 'dl/openface_nn4.small2.v1.t7', 'path to OpenCVs deep learning face embedding model')
flags.DEFINE_string( 'RECOGNIZER', 'dl/output/recognizer.pickle', 'path to model trained to recognize faces')

flags.DEFINE_string( 'LE2', 'dl/output/le2.pickle', 'path to label encoder')
flags.DEFINE_string( 'INPUT', args['video'], 'path to input video')
flags.DEFINE_string( 'OUTPUT', 'dl/capture_output', 'path to input video')

flags.DEFINE_integer( 'SKIP', 1, 'frames to skip before applying face detection')
flags.DEFINE_boolean( 'CONFIDENCE', 0.5, 'minimum probability to filter weak detections' )

FLAGS = flags.FLAGS


def main():
    assert FLAGS.MODE in ('train', 'valid', 'demo')
    print('THis is main')
    if FLAGS.MODE == 'demo':

        demo(FLAGS.DETECTOR, FLAGS.MODEL, FLAGS.LE, FLAGS.EMBEDDING_MODEL, FLAGS.RECOGNIZER, FLAGS.LE2, FLAGS.INPUT, FLAGS.OUTPUT, FLAGS.SKIP, FLAGS.CONFIDENCE, FLAGS.checkpoint_dir, FLAGS.show_box)

    elif FLAGS.MODE == 'train':
        train_model(FLAGS.train_data)
    elif FLAGS.MODE == 'valid':
        valid_model(FLAGS.checkpoint_dir, FLAGS.valid_data)


if __name__ == '__main__':
    main()