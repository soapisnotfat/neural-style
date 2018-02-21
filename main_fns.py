import argparse
from fast_neural_style.solver import Trainer, Stylizer


def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for fast_neural_style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=10, help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--batch-size", type=int, default=12, help="batch size for training, default is 4")
    train_arg_parser.add_argument("--dataset", type=str, default='./dataset', help="path to training dataset, the path should point to a folder containing another folder with all the training images")
    train_arg_parser.add_argument("--style-image", type=str, default="style-images/mosaic.jpg", help="path to style-image")
    train_arg_parser.add_argument("--save-model-dir", type=str, default='./models', help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--checkpoint-model-dir", type=str, default='./models', help="path to folder where checkpoints of trained models will be saved")
    train_arg_parser.add_argument("--image-size", type=int, default=256, help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument("--style-size", type=int, default=None, help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--seed", type=int, default=42, help="random seed for training")
    train_arg_parser.add_argument("--content-weight", type=float, default=1e5, help="weight for content-loss, default is 1e5")
    train_arg_parser.add_argument("--style-weight", type=float, default=1e10, help="weight for style-loss, default is 1e10")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,  help="learning rate, default is 1e-3")
    train_arg_parser.add_argument("--log-interval", type=int, default=500, help="number of images after which the training loss is logged, default is 500")
    train_arg_parser.add_argument("--checkpoint-interval", type=int, default=2000, help="number of batches after which a checkpoint of the trained model will be created")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content-image", type=str, required=True, help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None, help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output-image", type=str, default='./', help="path for saving the output image")
    eval_arg_parser.add_argument("--model", type=str, required=True, help="saved model to be used for stylizing the image")

    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        raise Exception("ERROR: specify either train or eval")

    if args.subcommand == "train":
        trainer = Trainer(args)
        trainer.validate()
    else:
        stylizer = Stylizer(args)
        stylizer.stylize()


if __name__ == "__main__":
    main()
