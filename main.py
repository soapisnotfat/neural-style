import argparse
import os
import time

import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from misc import progress_bar as pb
from PIL import Image
from model import TransformerNet, VGG16


class Trainer(object):
    def __init__(self, args):
        # dataset
        self.train_loader = None
        self.dataset = args.dataset
        self.style_image = args.style_image
        self.style_size = args.style_size

        # style
        self.gram_style = None
        self.vgg = None

        # model
        self.transformer = None
        self.optimizer = None
        self.criterion = None
        self.seed = args.seed

        # hyper-parameters
        self.image_size = args.image_size
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.epochs = args.epochs
        self.content_weight = args.content_weight
        self.style_weight = args.style_weight

        # general
        self.cuda = torch.cuda.is_available()
        self.log_interval = args.log_interval

        # tracking
        self.count = None
        self.agg_content_loss = None
        self.agg_style_loss = None

        # directory
        self.checkpoint_interval = args.checkpoint_interval
        self.save_model_dir = args.save_model_dir
        self.checkpoint_model_dir = args.checkpoint_model_dir

    def get_dataset(self):
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])

        train_dataset = datasets.ImageFolder(self.dataset, transform)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size)

    def get_style(self):
        # set up style
        style_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        style = load_image(self.style_image, size=self.style_size)
        style = style_transform(style)
        style = style.repeat(self.batch_size, 1, 1, 1)

        # set up feature extractor
        self.vgg = VGG16(requires_grad=False)

        if self.cuda:
            self.vgg.cuda()
            style = style.cuda()

        style_v = Variable(style)
        style_v = self.normalize_batch(style_v)
        features_style = self.vgg(style_v)
        self.gram_style = [self.gram_matrix(y) for y in features_style]

    def build_model(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.transformer = TransformerNet()
        self.optimizer = Adam(self.transformer.parameters(), self.lr)
        self.criterion = torch.nn.MSELoss()

        if self.cuda:
            torch.cuda.manual_seed(self.seed)
            self.transformer.cuda()

    @staticmethod
    def gram_matrix(y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram

    @staticmethod
    def normalize_batch(batch):
        # normalize using imageNet mean and std
        mean = batch.data.new(batch.data.size())
        std = batch.data.new(batch.data.size())
        mean[:, 0, :, :] = 0.485
        mean[:, 1, :, :] = 0.456
        mean[:, 2, :, :] = 0.406
        std[:, 0, :, :] = 0.229
        std[:, 1, :, :] = 0.224
        std[:, 2, :, :] = 0.225
        batch = torch.div(batch, 255.0)
        batch -= Variable(mean)
        batch = batch / Variable(std)
        return batch

    def check_paths(self, args):
        try:
            if not os.path.exists(self.save_model_dir):
                os.makedirs(self.save_model_dir)
            if args.checkpoint_model_dir is not None and not (os.path.exists(self.checkpoint_model_dir)):
                os.makedirs(self.checkpoint_model_dir)
        except OSError as e:
            raise Exception(e)

    def train(self, epoch):
        self.transformer.train()
        for batch_id, (x, _) in enumerate(self.train_loader):
            n_batch = len(x)
            self.count += n_batch
            self.optimizer.zero_grad()
            x = Variable(x)
            if self.cuda:
                x = x.cuda()

            y = self.transformer(x)

            y = self.normalize_batch(y)
            x = self.normalize_batch(x)

            features_y = self.vgg(y)
            features_x = self.vgg(x)

            content_loss = self.content_weight * self.criterion(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, self.gram_style):
                gm_y = self.gram_matrix(ft_y)
                style_loss += self.criterion(gm_y, gm_s[:n_batch, :, :])
            style_loss *= self.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            self.optimizer.step()

            self.agg_content_loss += content_loss.data[0]
            self.agg_style_loss += style_loss.data[0]

            pb(batch_id, len(self.train_loader), "content_loss: {:.6f} | style_loss: {:.6f} | total_loss: {:.6f}".format(self.agg_content_loss / (batch_id + 1), self.agg_style_loss / (batch_id + 1), (self.agg_content_loss + self.agg_style_loss) / (batch_id + 1)))
            self.checkpoint_save(batch_id, epoch)

    def checkpoint_save(self, batch_num, epoch):
        if self.checkpoint_model_dir is not None and (batch_num + 1) % self.checkpoint_interval == 0:
            self.transformer.eval()
            if self.cuda:
                self.transformer.cpu()
            checkpoint_model_filename = "checkpoint_epoch_" + str(epoch) + "_batch_id_" + str(batch_num + 1) + ".pth"
            checkpoint_model_path = os.path.join(self.checkpoint_model_dir, checkpoint_model_filename)
            torch.save(self.transformer.state_dict(), checkpoint_model_path)
            if self.cuda:
                self.transformer.cuda()

    def save(self):
        self.transformer.eval()
        if self.cuda:
            self.transformer.cpu()
        save_model_filename = "epoch_" + str(self.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
            self.content_weight) + "_" + str(self.style_weight) + ".model"
        save_model_path = os.path.join(self.save_model_dir, save_model_filename)
        torch.save(self.transformer.state_dict(), save_model_path)

        print("\nDone, trained model saved at", save_model_path)

    def validate(self):
        self.get_dataset()
        self.get_style()
        self.build_model()

        self.count = 0
        self.agg_content_loss = 0.
        self.agg_style_loss = 0.
        for e in range(self.epochs):
            print("Epoch {}".format(e + 1))
            self.train(e)
        self.save()


class Stylizer(object):
    def __init__(self, args):
        self.model = args.model
        self.content_image = args.content_image
        self.output_image = args.output_image
        self.content_scale = args.content_scale
        self.cuda = torch.cuda.is_available()

    def stylize(self):
        current_content_image = load_image(self.content_image, scale=self.content_scale)
        content_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        current_content_image = content_transform(current_content_image)
        current_content_image = current_content_image.unsqueeze(0)
        if self.cuda:
            current_content_image = current_content_image.cuda()
        current_content_image = Variable(current_content_image, volatile=True)

        style_model = TransformerNet()
        style_model.load_state_dict(torch.load(self.model))
        if self.cuda:
            style_model.cuda()
        output = style_model(current_content_image)
        if self.cuda:
            output = output.cpu()
        output_data = output.data[0]
        save_image(self.output_image, output_data)


def load_image(filename, size=None, scale=None):
    image = Image.open(filename)
    if size is not None:
        image = image.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        image = image.resize((int(image.size[0] / scale), int(image.size[1] / scale)), Image.ANTIALIAS)
    return image


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
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
    train_arg_parser.add_argument("--cuda", type=int, default=1, help="set it to 1 for running on GPU, 0 for CPU")
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
    eval_arg_parser.add_argument("--cuda", type=int, default=1, help="set it to 1 for running on GPU, 0 for CPU")

    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        raise Exception("ERROR: specify either train or eval")
    if args.cuda and not torch.cuda.is_available():
        raise Exception("ERROR: cuda is not available, try running on CPU")

    if args.subcommand == "train":
        trainer = Trainer(args)
        trainer.validate()
    else:
        stylizer = Stylizer(args)
        stylizer.stylize()


if __name__ == "__main__":
    main()
