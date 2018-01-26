from argparse import ArgumentParser
import chainer
from src.resnet import ResNet49Layers
from chainer.training import extensions

from src.dataset import LabelHandler
from src.data_loader import SortedImageDatasetBuilder
from tripletembedding.predictors.tripletnet import TripletNet
from tripletembedding.iterators import TripletIterator


def run_train():
    parser = ArgumentParser()
    parser.add_argument('--paths', type=str, nargs='+', required=True,
                        help='Root paths of folders that contain images and pascal voc files')
    parser.add_argument('--label_names', type=str, required=True,
                        help='Path to label names file')
    parser.add_argument('--training_splitsize', type=float, default=0.9,
                        help='Splitsize of training data')
    parser.add_argument('--batchsize', type=int, default=20,
                        help='Learning minibatch size')
    parser.add_argument('--epoch', type=int, default=10,
                        help='Numbers of epochs to train')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID, negative value indicates CPU')
    parser.add_argument('--out', default='trainer_output',
                        help='Output directory of trainer')
    parser.add_argument('--val_batchsize', type=int, default=250,
                        help='Validation minibatch size')
    parser.add_argument('--pretrained_model', type=str, default=None)
    args = parser.parse_args()

    tn = TripletNet(cnn=ResNet49Layers, kwargs={'pretrained_model': args.pretrained_model})
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        tn.to_gpu()

    label_handler = LabelHandler(args.label_names)
    builder = SortedImageDatasetBuilder(args.paths, label_handler)
    train_dataset, val_dataset = builder.get_sorted_image_dataset_split(args.training_splitsize)

    train_iter = TripletIterator(train_dataset, args.batchsize)
    val_iter = TripletIterator(val_dataset, args.val_batchsize)

    # optimizer
    learning_rate = 0.01
    momentum = 0.9
    optimizer = chainer.optimizers.MomentumSGD(learning_rate, momentum)
    optimizer.setup(tn)

    # trainer
    updater = chainer.training.updater.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'), args.out)

    trainer.extend(extensions.LogReport())
    trainer.extend(chainer.training.extensions.ProgressBar(update_interval=10))

    print('@A')
    trainer.run()

    # save model
    output_file_path = '{0}/triplet_net_{1}_{2}.model'.format(args.out, args.batchsize, args.epoch)
    chainer.serializers.save_npz(output_file_path, tn)
