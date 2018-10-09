import dlib
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train_xml', metavar='str', type=str, required=True)
    parser.add_argument('--output', metavar='str', type=str, required=True,
                        default='./tmp/train_models/hog_features.svm')
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('-c', metavar='INT', type=int, default=5)
    parser.add_argument('--num_threads', metavar='INT', type=int, default=4)
    args = parser.parse_args()

    options = dlib.simple_object_detector_training_options()
    options.add_left_right_image_flips = args.flip
    options.C = args.c
    options.num_threads = args.num_threads
    options.be_verbose = True
    dlib.train_simple_object_detector(args.train_xml, args.output, options)
