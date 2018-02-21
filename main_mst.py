import torch

from multi_style_transfer.solver import Trainer, Optimizer, Evaluator

import argparse


parser = argparse.ArgumentParser(description="parser for PyTorch-Style-Transfer")
subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

# training args
train_arg = subparsers.add_parser("train", help="parser for training arguments")
train_arg.add_argument("--ngf", type=int, default=128, help="number of generator filter channels, default 128")
train_arg.add_argument("--epochs", type=int, default=2, help="number of training epochs, default is 2")
train_arg.add_argument("--batch-size", type=int, default=4, help="batch size for training, default is 4")
train_arg.add_argument("--dataset", type=str, default="./dataset/", help="path to training dataset, the path should point to a folder containing another folder with all the training images")
train_arg.add_argument("--style-folder", type=str, default="style-images/", help="path to style-folder")
train_arg.add_argument("--vgg-model-dir", type=str, default="multi_style_transfer/VGGs/", help="path to folder where pretrained VGG model will be saved.")
train_arg.add_argument("--save-model-dir", type=str, default="multi_style_transfer/models/", help="path to folder where trained model will be saved.")
train_arg.add_argument("--image-size", type=int, default=256, help="size of training images, default is 256 X 256")
train_arg.add_argument("--style-size", type=int, default=512, help="size of style-image, default is the original size of style image")
train_arg.add_argument("--cuda", type=int, default=1, help="set it to 1 for running on GPU, 0 for CPU")
train_arg.add_argument("--seed", type=int, default=42, help="random seed for training")
train_arg.add_argument("--content-weight", type=float, default=1.0, help="weight for content-loss, default is 1.0")
train_arg.add_argument("--style-weight", type=float, default=5.0, help="weight for style-loss, default is 5.0")
train_arg.add_argument("--lr", type=float, default=1e-3, help="learning rate, default is 0.001")
train_arg.add_argument("--log-interval", type=int, default=500, help="number of images after which the training loss is logged, default is 500")
train_arg.add_argument("--resume", type=str, default=None, help="resume if needed")

# optim args (Gatys CVPR 2016)
optim_arg = subparsers.add_parser("optim", help="parser for optimization arguments")
optim_arg.add_argument("--iters", type=int, default=500, help="number of training iterations, default is 500")
optim_arg.add_argument("--content-image", type=str, default="test.jpg", help="path to content image you want to stylize")
optim_arg.add_argument("--style-image", type=str, default="style-images/candy.jpg", help="path to style-image")
optim_arg.add_argument("--content-size", type=int, default=512, help="factor for scaling down the content image")
optim_arg.add_argument("--style-size", type=int, default=512, help="size of style-image, default is the original size of style image")
optim_arg.add_argument("--output-image", type=str, default="output.jpg", help="path for saving the output image")
train_arg.add_argument("--vgg_model_dir", type=str, default="multi_style_transfer/VGGs/", help="path to folder where pretrained VGG model will be saved.")
optim_arg.add_argument("--cuda", type=int, default=1, help="set it to 1 for running on GPU, 0 for CPU")
optim_arg.add_argument("--content-weight", type=float, default=1.0, help="weight for content-loss, default is 1.0")
optim_arg.add_argument("--style-weight", type=float, default=5.0, help="weight for style-loss, default is 5.0")
optim_arg.add_argument("--lr", type=float, default=1e1, help="learning rate, default is 0.001")
optim_arg.add_argument("--log-interval", type=int, default=50, help="number of images after which the training loss is logged, default is 50")

# evaluation args
eval_arg = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
eval_arg.add_argument("--ngf", type=int, default=128, help="number of generator filter channels, default 128")
eval_arg.add_argument("--content-image", type=str, required=True, help="path to content image you want to stylize")
eval_arg.add_argument("--style-image", type=str, default="style-images/candy.jpg", help="path to style-image")
eval_arg.add_argument("--content-size", type=int, default=512, help="factor for scaling down the content image")
eval_arg.add_argument("--style-size", type=int, default=512, help="size of style-image, default is the original size of style image")
eval_arg.add_argument("--style-folder", type=str, default="style-images/", help="path to style-folder")
eval_arg.add_argument("--output-image", type=str, default="output.jpg", help="path for saving the output image")
eval_arg.add_argument("--model", type=str, required=True, help="saved model to be used for stylizing the image")
eval_arg.add_argument("--cuda", type=int, default=1, help="set it to 1 for running on GPU, 0 for CPU")
eval_arg.add_argument("--vgg-model-dir", type=str, default="multi_style_transfer/VGGs/", help="directory for vgg, if model is not present in the directory it is downloaded")

# demo
demo_arg = subparsers.add_parser("demo", help="parser for evaluation/stylizing arguments")
demo_arg.add_argument("--style-folder", type=str, default="style-images/", help="path to style-folder")
demo_arg.add_argument("--style-size", type=int, default=512, help="size of style-image, default is the original size of style image")
demo_arg.add_argument("--cuda", type=int, default=1, help="set it to 1 for running on GPU, 0 for CPU")
demo_arg.add_argument("--record", type=int, default=0, help="set it to 1 for recording into video file")
demo_arg.add_argument("--model", type=str, required=True, help="saved model to be used for stylizing the image")
demo_arg.add_argument("--ngf", type=int, default=128, help="number of generator filter channels, default 128")
demo_arg.add_argument("--demo-size", type=int, default=480, help="demo window height, default 480")

args = parser.parse_args()


def main():
    # figure out the experiments type
    if args.subcommand is None:
        raise ValueError("ERROR: specify the experiment type")
    if args.cuda and not torch.cuda.is_available():
        args.cuda = False
        print("Since GPU is not available, model will run on CPU")

    if args.subcommand == "train":
        # Training the model
        trainer = Trainer(args)
        trainer.train()

    elif args.subcommand == 'eval':
        # Test the pre-trained model
        evaluator = Evaluator(args)
        evaluator.evaluate()

    elif args.subcommand == 'optim':
        # Gatys et al. using optimization-based approach
        optimizer = Optimizer(args)
        optimizer.optimize()

    else:
        raise ValueError('Unknown experiment type')


if __name__ == "__main__":
    main()
