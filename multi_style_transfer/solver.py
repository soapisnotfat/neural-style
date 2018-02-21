import os
import sys
import time

import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.utils.serialization import load_lua

from misc import progress_bar
from PIL import Image
from multi_style_transfer.model import TransformerNet, VGG16


class Trainer(object):
    def __init__(self, args):
        # argument
        self.args = args

        # dataloader
        self.dataloader = None

    def build_dataloader(self):
        self.check_paths()
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        if self.args.cuda:
            torch.cuda.manual_seed(self.args.seed)
            kwargs = {'num_workers': 0, 'pin_memory': False}
        else:
            kwargs = {}

        transform = transforms.Compose([transforms.Resize(self.args.image_size),
                                        transforms.CenterCrop(self.args.image_size),
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda i: i.mul(255))])
        train_dataset = datasets.ImageFolder(self.args.dataset, transform)
        self.dataloader = DataLoader(train_dataset, batch_size=self.args.batch_size, **kwargs)

    def save_model(self, style_model):
        style_model.eval()
        style_model.cpu()
        save_model_filename = "Final_epoch_" + str(self.args.epochs) + "_" + str(time.ctime()).replace(' ', '_') \
                              + "_" + str(self.args.content_weight) + "_" + str(self.args.style_weight) + ".model"
        save_model_path = os.path.join(self.args.save_model_dir, save_model_filename)
        torch.save(style_model.state_dict(), save_model_path)

        print("\nDone, trained model saved at", save_model_path)

    def train(self):
        # build dataloader
        self.build_dataloader()

        # build model
        style_model = TransformerNet(ngf=self.args.ngf)

        if self.args.resume is not None:
            print('Resuming, initializing using weight from {}.'.format(self.args.resume))
            style_model.load_state_dict(torch.load(self.args.resume))

        optimizer = Adam(style_model.parameters(), self.args.lr)
        mse_loss = torch.nn.MSELoss()

        vgg = VGG16()
        init_vgg16(self.args.vgg_model_dir)
        vgg.load_state_dict(torch.load(os.path.join(self.args.vgg_model_dir, "vgg16.weight")))

        if self.args.cuda:
            style_model.cuda()
            vgg.cuda()
            cudnn.benchmark = True

        style_loader = StyleLoader(self.args.style_folder, self.args.style_size)

        for e in range(self.args.epochs):
            style_model.train()
            agg_content_loss = 0.
            agg_style_loss = 0.
            count = 0

            print("===> EPOCH: {}/{}".format(e + 1, self.args.epochs))

            for num_batch, (x, _) in enumerate(self.dataloader):
                n_batch = len(x)
                count += n_batch
                optimizer.zero_grad()
                x = Variable(preprocess_batch(x))
                if self.args.cuda:
                    x = x.cuda()

                style_v = style_loader.get(num_batch)
                style_model.set_target(style_v)

                style_v = subtract_image_net_mean_batch(style_v)
                features_style = vgg(style_v)
                gram_style = [gram_matrix(y) for y in features_style]

                y = style_model(x)
                xc = Variable(x.data.clone(), volatile=True)

                y = subtract_image_net_mean_batch(y)
                xc = subtract_image_net_mean_batch(xc)

                features_y = vgg(y)
                features_xc = vgg(xc)

                f_xc_c = Variable(features_xc[1].data, requires_grad=False)

                content_loss = self.args.content_weight * mse_loss(features_y[1], f_xc_c)

                style_loss = 0.
                for m in range(len(features_y)):
                    gram_y = gram_matrix(features_y[m])
                    gram_s = Variable(gram_style[m].data, requires_grad=False).repeat(self.args.batch_size, 1, 1, 1)
                    style_loss += self.args.style_weight * mse_loss(gram_y, gram_s[:n_batch, :, :])

                total_loss = content_loss + style_loss
                total_loss.backward()
                optimizer.step()

                agg_content_loss += content_loss.data[0]
                agg_style_loss += style_loss.data[0]

                progress_bar(num_batch, len(self.dataloader), "content: {:.6f} | style: {:.6f} | total: {:.6f}".format(
                    agg_content_loss / (num_batch + 1), agg_style_loss / (num_batch + 1),
                    (agg_content_loss + agg_style_loss) / (num_batch + 1)))

                if (num_batch + 1) % self.args.log_interval == 0:
                    # save model
                    style_model.eval()
                    style_model.cpu()
                    save_model_filename = "Epoch_" + str(e) + "iters_" + str(count) + "_" + str(time.ctime()).replace(
                        ' ',
                        '_') + "_" + str(
                        self.args.content_weight) + "_" + str(self.args.style_weight) + ".model"
                    save_model_path = os.path.join(self.args.save_model_dir, save_model_filename)
                    torch.save(style_model.state_dict(), save_model_path)
                    style_model.train()
                    style_model.cuda()
                    print("\nCheckpoint, trained model saved at", save_model_path)

        # save model
        self.save_model(style_model=style_model)

    def check_paths(self):
        try:
            if not os.path.exists(self.args.vgg_model_dir):
                os.makedirs(self.args.vgg_model_dir)
            if not os.path.exists(self.args.save_model_dir):
                os.makedirs(self.args.save_model_dir)
        except OSError as e:
            print(e)
            sys.exit(1)


class Optimizer(object):
    """
    Gatys et al. CVPR 2017
    ref: Image Style Transfer Using Convolutional Neural Networks
    """

    def __init__(self, args):
        self.args = args

        # images
        self.content_image = None
        self.style_image = None

        # model
        self.vgg = None

    def build_up(self):
        # load content image
        content_image = rgb_to_tensor(self.args.content_image, size=self.args.content_size, keep_asp=True)
        content_image = content_image.unsqueeze(0)
        content_image = Variable(preprocess_batch(content_image), requires_grad=False)
        self.content_image = subtract_image_net_mean_batch(content_image)

        # load style target
        style_image = rgb_to_tensor(self.args.style_image, size=self.args.style_size)
        style_image = style_image.unsqueeze(0)
        style_image = Variable(preprocess_batch(style_image), requires_grad=False)
        self.style_image = subtract_image_net_mean_batch(style_image)

        # setup vgg features
        self.vgg = VGG16()
        init_vgg16(self.args.vgg_model_dir)
        self.vgg.load_state_dict(torch.load(os.path.join(self.args.vgg_model_dir, "vgg16.weight")))

        # if using GPU
        if self.args.cuda:
            self.content_image = content_image.cuda()
            self.style_image = style_image.cuda()
            self.vgg.cuda()
            cudnn.benchmark = True

    def save_image(self, output):
        output = add_image_net_mean_batch(output)
        bgr_to_tensor(output.data[0], self.args.output_image, self.args.cuda)

    def get_features(self):
        features_content = self.vgg(self.content_image)
        f_xc_c = Variable(features_content[1].data, requires_grad=False)
        features_style = self.vgg(self.style_image)
        gram_style = [gram_matrix(y) for y in features_style]

        return f_xc_c, gram_style

    def optimize(self):
        # setup content, styles, and feature nets
        self.build_up()

        # get features
        f_xc_c, gram_style = self.get_features()

        # init optimizer
        output = Variable(self.content_image.data, requires_grad=True)
        optimizer = Adam([output], lr=self.args.lr)
        mse_loss = torch.nn.MSELoss()

        # optimizing the images
        for e in range(self.args.iters):
            image_net_clamp_batch(output, 0, 255)
            optimizer.zero_grad()
            features_y = self.vgg(output)
            content_loss = self.args.content_weight * mse_loss(features_y[1], f_xc_c)

            style_loss = 0.
            for m in range(len(features_y)):
                gram_y = gram_matrix(features_y[m])
                gram_s = Variable(gram_style[m].data, requires_grad=False)
                style_loss += self.args.style_weight * mse_loss(gram_y, gram_s)

            total_loss = content_loss + style_loss

            if (e + 1) % self.args.log_interval == 0:
                print(total_loss.data.cpu().numpy()[0])
            total_loss.backward()

            optimizer.step()

        # save the image
        self.save_image(output)


class Evaluator(object):
    def __init__(self, args):
        self.args = args

    def evaluate(self):
        content_image = rgb_to_tensor(self.args.content_image, size=self.args.content_size, keep_asp=True)
        content_image = content_image.unsqueeze(0)
        style = rgb_to_tensor(self.args.style_image, size=self.args.style_size)
        style = style.unsqueeze(0)
        style = preprocess_batch(style)

        style_model = TransformerNet(ngf=self.args.ngf)
        style_model.load_state_dict(torch.load(self.args.model))

        if self.args.cuda:
            style_model.cuda()
            content_image = content_image.cuda()
            style = style.cuda()

        style_v = Variable(style, volatile=True)

        content_image = Variable(preprocess_batch(content_image), volatile=True)
        style_model.setTarget(style_v)

        output = style_model(content_image)
        bgr_to_tensor(output.data[0], self.args.output_image, self.args.cuda)

    def fast_evaluate(self, basedir, contents, idx=0):
        # basedir to save the data
        style_model = TransformerNet(ngf=self.args.ngf)
        style_model.load_state_dict(torch.load(self.args.model))
        style_model.eval()
        if self.args.cuda:
            style_model.cuda()

        style_loader = StyleLoader(self.args.style_folder, self.args.style_size,
                                   cuda=self.args.cuda)

        for content_image in contents:
            idx += 1
            content_image = rgb_to_tensor(content_image, size=self.args.content_size, keep_asp=True).unsqueeze(0)
            if self.args.cuda:
                content_image = content_image.cuda()
            content_image = Variable(preprocess_batch(content_image), volatile=True)

            for isx in range(style_loader.size()):
                style_v = Variable(style_loader.get(isx).data, volatile=True)
                style_model.setTarget(style_v)
                output = style_model(content_image)
                filename = os.path.join(basedir, "{}_{}.png".format(idx, isx + 1))
                bgr_to_tensor(output.data[0], filename, self.args.cuda)
                print(filename)


class StyleLoader:
    def __init__(self, style_folder, style_size, cuda=True):
        self.folder = style_folder
        self.style_size = style_size
        self.files = os.listdir(style_folder)
        self.cuda = cuda

    def get(self, i):
        idx = i % len(self.files)
        file_path = os.path.join(self.folder, self.files[idx])
        style = rgb_to_tensor(file_path, self.style_size)
        style = style.unsqueeze(0)
        style = preprocess_batch(style)
        if self.cuda:
            style = style.cuda()
        style_v = Variable(style, requires_grad=False)
        return style_v

    def size(self):
        return len(self.files)


def rgb_to_tensor(filename, size=None, scale=None, keep_asp=False):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)

    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img


def tensor_save_rgb(tensor, filename, cuda=False):
    if cuda:
        img = tensor.clone().cpu().clamp(0, 255).numpy()
    else:
        img = tensor.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)


def bgr_to_tensor(tensor, filename, cuda=False):
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r, g, b))
    tensor_save_rgb(tensor, filename, cuda)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def subtract_image_net_mean_batch(batch):
    """Subtract ImageNet mean pixel-wise from a BGR image."""
    tensor_type = type(batch.data)
    mean = tensor_type(batch.data.size())
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    return batch - Variable(mean)


def add_image_net_mean_batch(batch):
    """Add ImageNet mean pixel-wise from a BGR image."""
    tensor_type = type(batch.data)
    mean = tensor_type(batch.data.size())
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    return batch + Variable(mean)


def image_net_clamp_batch(batch, low, high):
    batch[:, 0, :, :].data.clamp_(low - 103.939, high - 103.939)
    batch[:, 1, :, :].data.clamp_(low - 116.779, high - 116.779)
    batch[:, 2, :, :].data.clamp_(low - 123.680, high - 123.680)


def preprocess_batch(batch):
    batch = batch.transpose(0, 1)
    (r, g, b) = torch.chunk(batch, 3)
    batch = torch.cat((b, g, r))
    batch = batch.transpose(0, 1)
    return batch


def init_vgg16(model_folder):
    """load the vgg16 model feature"""
    if not os.path.exists(os.path.join(model_folder, 'vgg16.weight')):
        if not os.path.exists(os.path.join(model_folder, 'vgg16.t7')):
            os.system(
                'wget http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/vgg16.t7 -O ' + os.path.join(
                    model_folder, 'vgg16.t7'))
        vgglua = load_lua(os.path.join(model_folder, 'vgg16.t7'))
        vgg = VGG16()
        for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
            dst.data[:] = src
        torch.save(vgg.state_dict(), os.path.join(model_folder, 'vgg16.weight'))
