import os
import time

import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from misc import progress_bar
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
        style = self.load_image(self.style_image, size=self.style_size)
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
        (batches, channel, height, width) = y.size()
        features = y.view(batches, channel, width * height)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (channel * height * width)
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

    @staticmethod
    def load_image(filename, size=None, scale=None):
        image = Image.open(filename)
        if size is not None:
            image = image.resize((size, size), Image.ANTIALIAS)
        elif scale is not None:
            image = image.resize((int(image.size[0] / scale), int(image.size[1] / scale)), Image.ANTIALIAS)
        return image

    def check_paths(self, args):
        try:
            if not os.path.exists(self.save_model_dir):
                os.makedirs(self.save_model_dir)
            if args.checkpoint_model_dir is not None and not (os.path.exists(self.checkpoint_model_dir)):
                os.makedirs(self.checkpoint_model_dir)
        except OSError as e:
            raise Exception(e)

    def train(self, epoch):
        self.transformer.train()  # turn on train mode

        for batch_id, (data, _) in enumerate(self.train_loader):
            # get data and target
            data = Variable(data.cuda() if self.cuda else data)
            target = self.transformer(data)
            data = self.normalize_batch(data)
            target = self.normalize_batch(target)

            # calculate content loss
            target_feature = self.vgg(target)
            data_feature = self.vgg(data)
            content_loss = self.content_weight * self.criterion(target_feature.relu2_2, data_feature.relu2_2)

            # calculate style loss
            style_loss = 0.
            for current_target_feature, current_gram_style in zip(target_feature, self.gram_style):
                current_target_feature = self.gram_matrix(current_target_feature)
                style_loss += self.criterion(current_target_feature, current_gram_style[:len(data), :, :])
            style_loss *= self.style_weight

            # do the backpropagation
            total_loss = content_loss + style_loss
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            self.agg_content_loss += content_loss.data[0]
            self.agg_style_loss += style_loss.data[0]

            progress_bar(batch_id, len(self.train_loader), "content_loss: {:.6f} | style_loss: {:.6f} | total_loss: {:.6f}".format(self.agg_content_loss / (batch_id + 1), self.agg_style_loss / (batch_id + 1), (self.agg_content_loss + self.agg_style_loss) / (batch_id + 1)))
            self.checkpoint_save(batch_id, epoch)

    def checkpoint_save(self, batch_num, epoch):

        if (batch_num + 1) % self.checkpoint_interval == 0:
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

        self.agg_content_loss = 0.
        self.agg_style_loss = 0.

        if not os.path.exists(self.checkpoint_model_dir):
            os.makedirs(self.checkpoint_model_dir)

        for e in range(self.epochs):
            print("Epoch {}".format(e + 1))
            self.train(e)
        self.save()


class Stylizer(object):
    """
    The Stylizer that transforms content images to one in expected style
    """
    def __init__(self, args):
        self.model = args.model
        self.content_image = args.content_image
        self.output_image = args.output_image
        self.content_scale = args.content_scale
        self.cuda = torch.cuda.is_available()

    @staticmethod
    def save_image(filename, data):
        img = data.clone().clamp(0, 255).numpy()
        img = img.transpose(1, 2, 0).astype("uint8")
        img = Image.fromarray(img)
        img.save(filename)

    @staticmethod
    def load_image(filename, size=None, scale=None):
        out_image = Image.open(filename)  # open the file

        # apply some potential changes
        if size is not None:
            out_image = out_image.resize((size, size), Image.ANTIALIAS)
        if scale is not None:
            out_image = out_image.resize((int(out_image.size[0] / scale), int(out_image.size[1] / scale)), Image.ANTIALIAS)

        return out_image

    def stylize(self):
        # load content image
        current_content_image = self.load_image(self.content_image, scale=self.content_scale)
        content_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.mul(255))])
        current_content_image = content_transform(current_content_image).unsqueeze(0)

        # cast content images to Tensor
        current_content_image = (current_content_image.cuda() if self.cuda else current_content_image)
        current_content_image = Variable(current_content_image, volatile=True)

        # load transformer model
        style_model = TransformerNet()
        style_model.load_state_dict(torch.load(self.model))
        if self.cuda:
            style_model.cuda()

        # transforming
        output = style_model(current_content_image)

        # save the image
        output = (output.cpu() if self.cuda else output)
        output_data = output.data[0]
        self.save_image(self.output_image + self.content_image.split('.')[0] + '-out.jpg', output_data)
