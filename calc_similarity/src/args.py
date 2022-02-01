import argparse


def argument_parser():
    parser = argparse.ArgumentParser(description="Run SimGNN.")

    parser.add_argument(
        "--train-dir",
        nargs="*",
        default=["./dataset/SICK/train/"],
        help="Folder with training graph pair jsons.",
    )

    parser.add_argument(
        "--test-dir",
        nargs="*",
        default=["./dataset/SICK/test/"],
        help="Folder with testing graph pair jsons.",
    )

    parser.add_argument(
        "--valid-dir",
        nargs="*",
        default=["./dataset/SICK/validation/"],
        help="Folder with vaidation graph pair jsons.",
    )

    parser.add_argument(
        "--tensorboard-dir",
        default="./TensorBoard/default/",
        help="Folder with TensorBoard Logs.",
    )

    parser.add_argument(
        "--use-sagpool",
        action="store_true",
        help="Boolean whether to use SAGPooling or not. Default is False",
    )

    parser.add_argument(
        "--use-cos",
        action="store_true",
        help="Boolean whether to use CosineSimilarity or not. Default is False",
    )

    parser.add_argument(
        "--use-sage",
        action="store_true",
        help="Boolean whether to use GrapgSAGE or not. Default is False",
    )

    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=500,
        help="Number of training epochs. Default is 500.",
    )

    parser.add_argument(
        "--early-stop",
        type=int,
        default=10,
        help="Number of early stopping count. Default is 10.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=128,
        help="Number of graph pairs per batch. Default is 128.",
    )

    parser.add_argument(
        "-l",
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate. Default: 0.001.",
    )

    parser.add_argument(
        "--input-dim",
        type=int,
        default=300,
        help="Size of each node feature. Default: 300",
    )

    parser.add_argument(
        "--channel-1",
        type=int,
        default=128,
        help="Size of 1st convolution output. Default: 128",
    )

    parser.add_argument(
        "--channel-2",
        type=int,
        default=64,
        help="Size of 2nd convolution output. Default: 64",
    )

    parser.add_argument(
        "--channel-3",
        type=int,
        default=32,
        help="Size of 3rd convolution output. Default: 32",
    )

    parser.add_argument(
        "-d", "--dropout", type=float, default=0.5, help="Dropout. Default: 0.5"
    )

    parser.add_argument(
        "--save-path", type=str, default=None, help="Where to save the trained model"
    )

    parser.add_argument(
        "--load-path", type=str, default=None, help="Load a pretrained model"
    )

    return parser.parse_args()
