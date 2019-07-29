# high l2 - works
# gen a two image
# sndisz two image
# self attention later layers
# no unet structure in gen a -- gen b
# add noise to gen
# create an image extractor generator for interpolation mode
# effect of batch size
# two separate discriminators
# Generates feature maps for target pose
# add interp flag to gens
# Generator initialization kaiming  refer https://github.com/iwtw/pytorch-TP-GAN/blob/master/layers.py
# Use coord conv to infer current pose rather than a template
# Spectral norm in Generator sagan

import os
import time
import datetime
import glob
from torchvision.utils import save_image
from torchvision import transforms as T
from PIL import Image
from data_loader import get_loader
from model import GenB, GenA, GenC, GenX, DisZ, GeneratorUNet, SNDisZ
from resources.logger import Logger
from config import export_config
from solver_utils import *
from scipy.misc import imsave
import os.path as osp
from resources.utils import ensure_dir_exists


# TODO: clean all symbol names : T and T_im arbitrary


class Solver(object):
    """Solver for training and testing"""

    def __init__(self, cfg):
        """Initialize configurations."""

        self.cfg = cfg

        device_id = cfg.params.cuda_device_id

        if 'cuda' in device_id and torch.cuda.is_available():
            self.device = torch.device(device_id)
        else:
            self.device = torch.device('cpu')

        print("Training on device {}".format(self.device))

        self.data_loader, self.num_classes = get_loader(self.cfg)

        print("Dataset loaded with {} classes".format(self.num_classes))

        self.use_tensorboard = cfg.flags.USE_TENSORBOARD
        if self.use_tensorboard:
            self.build_tensorboard()

        self.build_model()
        if self.cfg.train_param.LOAD_NUMBER_POSE:
            self.load_pose_images_nums()
        else:
            self.load_pose_images()

        self.loaded_samples = False
        self.loaded_interp = False
        self.loaded_result = False

    def build_model(self):
        num_classes = self.num_classes

        img_size = self.cfg.model_param.img_size
        beta1 = self.cfg.optimizer_param.beta1
        beta2 = self.cfg.optimizer_param.beta2
        g_lr = self.cfg.optimizer_param.g_lr
        d_lr = self.cfg.optimizer_param.d_lr
        chosen_gen = self.cfg.model_param.gen
        chosen_dis = self.cfg.model_param.dis

        """Create a generator and a discriminator."""
        if chosen_gen == 'genb':
            ChosenGen = GenB
        elif chosen_gen == "gena":
            ChosenGen = GenA
        elif chosen_gen == "genc":
            ChosenGen = GenC
        elif chosen_gen == "genx":
            ChosenGen = GenX
        elif chosen_gen == "unet":
            ChosenGen = GeneratorUNet
        else:
            raise ValueError('Chosen generator not found.')

        if chosen_dis == "sn":
            ChosenDis = SNDisZ
        else:
            ChosenDis = DisZ

        # todo clean device in chosen gen
        self.G = ChosenGen(self.device)
        self.D = ChosenDis(img_size, num_classes)

        if self.cfg.flags.DATA_PARALLEL:
            device_ids = self.cfg.params.parallel_device_id
            device_ids_list = ['cuda:{}'.format(i) for i in device_ids.split(',')]
            torch.nn.DataParallel(self.G, device_ids=device_ids_list, output_device=self.device)

        self.g_opt = torch.optim.Adam(self.G.parameters(), g_lr, [beta1, beta2])
        self.d_opt = torch.optim.Adam(self.D.parameters(), d_lr, [beta1, beta2])

        print_network(self.D, 'D')
        print_network(self.G, 'G')

        self.D.to(self.device)
        self.G.to(self.device)

        if self.cfg.flags.LOADED_CONFIG:
            self.restore_model()

    def restore_model(self):
        """Restore the trained generator and discriminator."""

        current_iter = self.cfg.train_param.global_iter

        print('Loading the trained models from step {}...'.format(current_iter))

        # load models
        g_path = osp.join(self.cfg.dirs.model_save_dir, '{}-G.ckpt'.format(current_iter))
        d_path = osp.join(self.cfg.dirs.model_save_dir, '{}-D.ckpt'.format(current_iter))

        self.G.load_state_dict(torch.load(g_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(d_path, map_location=lambda storage, loc: storage))

        # load optimizers
        g_opt_path = osp.join(self.cfg.dirs.model_save_dir, '{}-G_opt.pt'.format(current_iter))
        d_opt_path = osp.join(self.cfg.dirs.model_save_dir, '{}-D_opt.pt'.format(current_iter))

        self.g_opt.load_state_dict(torch.load(g_opt_path))
        self.d_opt.load_state_dict(torch.load(d_opt_path))

    def save_current_state(self, itr):
        """Saves the current state of model training"""

        # save models
        g_path = osp.join(self.cfg.dirs.model_save_dir, '{}-G.ckpt'.format(itr + 1))
        d_path = osp.join(self.cfg.dirs.model_save_dir, '{}-D.ckpt'.format(itr + 1))

        torch.save(self.G.state_dict(), g_path)
        torch.save(self.D.state_dict(), d_path)

        # save optimizer
        g_opt_path = osp.join(self.cfg.dirs.model_save_dir, '{}-G_opt.pt'.format(itr + 1))
        d_opt_path = osp.join(self.cfg.dirs.model_save_dir, '{}-D_opt.pt'.format(itr + 1))

        torch.save(self.g_opt.state_dict(), g_opt_path)
        torch.save(self.d_opt.state_dict(), d_opt_path)

        # export config
        self.cfg.train_param.global_iter = itr + 1
        export_config(self.cfg)

        print('Saved training state')

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        self.logger = Logger(self.cfg.dirs.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_opt.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_opt.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_opt.zero_grad()
        self.d_opt.zero_grad()

    def load_pose_images(self):
        img_size = self.cfg.model_param.img_size

        transform = T.Compose([T.Resize(img_size),
                               T.ToTensor()])

        cwd = os.getcwd()
        os.chdir(self.cfg.dirs.pose_img_dir)

        self.pose_images = {}
        for filename in glob.glob("*.jpg"):
            pose = int(filename[:3])
            self.pose_images[pose] = transform(Image.open(osp.join(os.getcwd(), filename)))

        # print('poses', self.pose_images.keys())
        os.chdir(cwd)

    def load_pose_images_nums(self):
        img_size = self.cfg.model_param.img_size

        def get_im_sized_pose(pose_num):
            return torch.ones(1, img_size, img_size) * pose_num

        train_poses = self.cfg.train_param.poses
        scale_fact = self.cfg.train_param.scale_fact_pose
        if train_poses[-1] == 180:
            rescale = lambda x: (x / 180.0 * 2 - 1) / scale_fact
        elif train_poses[-1] == 90:
            rescale = lambda x: x / 90.0 / 10.0 / scale_fact
        self.pose_images = {}
        for train_pose in train_poses:
            self.pose_images[train_pose] = torch.ones(1, img_size, img_size) * rescale(train_pose)

        a = 0

    def get_all_templates(self, batch_size):
        """Generate target domain labels for debugging and testing."""
        labels = []
        pose_values = list(self.pose_images.keys())
        pose_values.sort()
        print(pose_values)

        for pose in pose_values:
            if pose in self.cfg.train_param.poses:
                pose_img = self.pose_images[pose]
                # Repeat it across the batch!
                pose_img = pose_img.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(self.device)
                labels.append(pose_img)

        return labels

    def print_log(self, loss, itr, start_time):
        """ Print Progress of training"""
        et = time.time() - start_time
        et = str(datetime.timedelta(seconds=et))[:-7]
        log = "Elapsed [{}], Iteration [{}/{}]".format(et, itr, self.cfg.train_param.max_iters)
        for tag, value in loss.items():
            log += ", {}: {:.4f}".format(tag, value)
        print(log)

        if self.use_tensorboard:
            for tag, value in loss.items():
                self.logger.scalar_summary(tag, value, itr)

    def calc_disc_loss(self, a_real, b_real, c_real, a_T, b_T, c_T, class_idx, loss_log, loss_log_prefix):
        batch_size = self.cfg.train_param.batch_size
        # TODO: Wgan GP loss -> might have errors
        #

        # # d_loss_real = -out_real.mean()
        # # d_loss_fake = out_fake.mean()
        #
        # # Any point on the line for gradient penalty
        # eps = torch.rand(batch_size, 1, 1, 1).to(self.device)
        # b_hat = (eps * b_real.data + (1 - eps) * b_fake.data)
        # b_hat = Variable(b_hat, requires_grad=True)
        #
        # # GP for D, y_poseT is just a conditioning variable
        # out_src, _ = self.D(a_real, b_hat, c_real, a_T, b_T, c_T)
        #
        # d_loss_gp = self.gradient_penalty(out_src, b_hat, self.device)

        # a + c --> b
        d_out_real, out_cls_real = self.D(a_real, b_real, c_real, a_T, b_T, c_T)

        b_fake = self.G(a_real, c_real, a_T, b_T, c_T)
        d_out_fake, _ = self.D(a_real, b_fake.detach(), c_real, a_T, b_T, c_T)

        d_loss_cls_real = self.ce_loss(out_cls_real, class_idx.long())

        d_loss_real = self.relu_loss(1.0 - d_out_real).mean()
        d_loss_fake = self.relu_loss(1.0 + d_out_fake).mean()

        d_loss_gp = 0

        lambda_gp = self.cfg.model_param.lambda_gp
        lambda_cls = self.cfg.model_param.lambda_cls

        # L_real + L_fake + l_GP + l_classification
        d_loss = d_loss_real + d_loss_fake + \
                 lambda_gp * d_loss_gp + lambda_cls * d_loss_cls_real

        # Logging.

        loss_log[loss_log_prefix + 'real'] = d_loss_real.item()
        loss_log[loss_log_prefix + 'fake'] = d_loss_fake.item()
        # loss_log[loss_log_prefix + 'lGP'] = lambda_gp * d_loss_gp.item()
        loss_log[loss_log_prefix + 'cls'] = lambda_cls * d_loss_cls_real.item()
        loss_log[loss_log_prefix + 'total'] = d_loss.item()

        return d_loss, loss_log

    def calc_gen_loss_ab(self, a_real, b_real, c_real, a_T, b_T, c_T, class_idx, loss_log, loss_log_prefix):
        # Original pose to target pose.
        # self.vis_tensor(a_real)
        # self.vis_tensor(c_real)
        # self.vis_tensor(b_T, denorm_ar=False)
        # self.vis_tensor(b_real)

        b_fake = self.G(a_real, c_real, a_T, b_T, c_T)

        # Fake Image conditioned on pose loss
        out_src, out_cls = self.D(a_real, b_fake, c_real, a_T, b_T, c_T)
        g_pose_loss_fake = -out_src.mean()

        # Classification loss
        g_cls_loss = self.ce_loss(out_cls, class_idx.long())

        # Total Variational Regularisation
        tvr_loss = torch.sum(torch.abs(b_fake[:, :, 1:, :] - b_fake[:, :, :-1, :])) + \
                   torch.sum(torch.abs(b_fake[:, :, :, 1:] - b_fake[:, :, :, :-1]))
        tvr_loss = tvr_loss.mean()

        # L1 loss
        # l1_loss = torch.mean((b_fake - b_real) ** 2)
        l1_loss = torch.mean(torch.abs(b_fake - b_real))
        # l2_loss = torch.mean((b_fake - b_real)**2) ; lambda_l1 = 100 * lambda_l1

        lambda_cls = self.cfg.model_param.lambda_cls
        lambda_tvr = self.cfg.model_param.lambda_tvr
        lambda_l1 = self.cfg.model_param.lambda_l1
        g_loss = g_pose_loss_fake + lambda_tvr * tvr_loss + lambda_cls * g_cls_loss + lambda_l1 * l1_loss

        # Logging.
        loss_log[loss_log_prefix + 'fake'] = g_pose_loss_fake.item()
        loss_log[loss_log_prefix + 'tvr'] = lambda_tvr * tvr_loss.item()
        loss_log[loss_log_prefix + 'cls'] = lambda_cls * g_cls_loss.item()
        loss_log[loss_log_prefix + 'l1'] = lambda_l1 * l1_loss.item()
        loss_log[loss_log_prefix + 'total'] = g_loss.item()

        return g_loss, loss_log

    def train(self):

        print('Start training...')
        start_time = time.time()

        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.relu_loss = torch.nn.ReLU()

        data_loader = self.data_loader
        data_iter = iter(data_loader)

        # Learning rate cache for decaying.
        g_lr = self.cfg.optimizer_param.g_lr
        d_lr = self.cfg.optimizer_param.d_lr

        cur_iter = self.cfg.train_param.global_iter
        max_iter = self.cfg.train_param.max_iters

        model_save_step = self.cfg.train_param.model_save_step
        sample_step = self.cfg.train_param.sample_step
        log_step = self.cfg.train_param.log_step
        lr_update_step = self.cfg.train_param.lr_update_step
        lr_decay_rate = self.cfg.optimizer_param.lr_decay_rate
        critic_train_no = self.cfg.train_param.critic_train_no

        for itr in range(cur_iter, max_iter):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            try:
                class_idx, x, x_T, y, y_T, z, z_T = next(data_iter)
            except:
                data_iter = iter(data_loader)
                class_idx, x, x_T, y, y_T, z, z_T = next(data_iter)

            x = x.to(self.device)
            y = y.to(self.device)
            z = z.to(self.device)

            x_T_im = [self.pose_images[p] for p in x_T.numpy()]
            y_T_im = [self.pose_images[p] for p in y_T.numpy()]
            z_T_im = [self.pose_images[p] for p in z_T.numpy()]

            x_T_im = torch.stack(x_T_im, 0).to(self.device)
            y_T_im = torch.stack(y_T_im, 0).to(self.device)
            z_T_im = torch.stack(z_T_im, 0).to(self.device)

            class_idx = class_idx.to(self.device)

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #
            requires_grad(self.G, False)
            requires_grad(self.D, True)

            ### Notation
            # x --> y
            # y --> z
            # x+z encoded --> y
            loss_log = {}

            d_loss, loss_log = self.calc_disc_loss(x, y, z, x_T_im, y_T_im, z_T_im, class_idx, loss_log, 'D/')
            self.reset_grad()
            d_loss.backward(retain_graph=True)
            self.d_opt.step()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            if (itr + 1) % critic_train_no == 0:
                requires_grad(self.G, True)
                requires_grad(self.D, False)

                g_loss, loss_log = self.calc_gen_loss_ab(x, y, z, x_T_im, y_T_im, z_T_im, class_idx, loss_log, 'G/')

                self.reset_grad()
                g_loss.backward()
                self.g_opt.step()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #
            # Print out training information.
            if (itr + 1) % log_step == 0:
                self.print_log(loss_log, itr + 1, start_time)

            # Translate fixed images for debugging.
            if (itr + 1) % sample_step == 0:
                self.sample(itr)
                self.interpolate(itr)
                self.results(itr)

            # Save model checkpoints.
            if (itr + 1) % model_save_step == 0:
                self.save_current_state(itr)

            # Decay learning rates.
            if (itr + 1) % lr_update_step == 0:
                # TODO: check if this decay style is okay in GANs
                g_lr *= lr_decay_rate
                d_lr *= lr_decay_rate
                self.update_lr(g_lr, d_lr)
                print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def sample(self, itr):
        batch_size = self.cfg.sample_param.num_samples
        train_poses = self.cfg.train_param.poses

        if not self.loaded_samples:
            self.sample_a, sample_a_pnum, self.sample_c, sample_c_pnum = \
                self.data_loader.dataset.get_sample_imgs(train_poses, batch_size, self.device)
            pose_templates = self.pose_images

            stack_dev = lambda t: torch.stack(t).to(self.device)

            self.sample_a_T = stack_dev([pose_templates[pnum] for pnum in sample_a_pnum])
            self.sample_c_T = stack_dev([pose_templates[pnum] for pnum in sample_c_pnum])

            self.loaded_samples = True

        with torch.no_grad():
            self.G.eval()

            sample_im = [self.sample_a]

            for i in range(len(train_poses)):
                cur_pose_degree = train_poses[i]
                sample_b_T = self.pose_images[cur_pose_degree].unsqueeze(0).repeat(batch_size, 1, 1, 1).to(self.device)
                decoded_image = self.G(self.sample_a, self.sample_c, self.sample_a_T, sample_b_T, self.sample_c_T)

                sample_im.append(decoded_image)
            sample_im.append(self.sample_c)

        imgs = torch.cat(sample_im, dim=3)

        sample_path = osp.join(self.cfg.dirs.sample_dir, '{}-images.jpg'.format(itr + 1))
        save_image(denorm(imgs.data.cpu()), sample_path, nrow=1, padding=0)
        self.G.train()

        print('Saved Sample images for {} into {}...'.format(itr, sample_path))

    def results(self, itr):
        batch_size = self.cfg.sample_param.num_samples
        train_poses = self.cfg.train_param.poses
        result_dir = self.cfg.dirs.result_dir
        real_dir = osp.join(result_dir, 'real')
        fake_dir = osp.join(result_dir, 'fake')

        if not self.loaded_result:
            self.res_a, res_a_pnum, self.res_c, res_c_pnum = \
                self.data_loader.dataset.get_sample_imgs(train_poses, batch_size, self.device)
            pose_templates = self.pose_images

            stack_dev = lambda t: torch.stack(t).to(self.device)

            self.res_a_T = stack_dev([pose_templates[pnum] for pnum in res_a_pnum])
            self.res_c_T = stack_dev([pose_templates[pnum] for pnum in res_c_pnum])

            self.loaded_result = True

            ensure_dir_exists(real_dir)
            ensure_dir_exists(fake_dir)

        with torch.no_grad():
            self.G.eval()
            max_i = len(train_poses) - 1
            for i in range(max_i + 1):
                cur_pose_degree = train_poses[i]
                cur_pose_T = self.pose_images[cur_pose_degree].unsqueeze(0).repeat(batch_size, 1, 1, 1).to(self.device)

                decoded_image = self.G(self.res_a, self.res_c, self.res_a_T, cur_pose_T,
                                       self.res_c_T)

                dec = denorm(decoded_image.data.cpu()).numpy()[0].transpose(1, 2, 0)
                dec_im_name = osp.join(fake_dir, 'itr{}-pose{}.jpg'.format(itr + 1, i))
                imsave(dec_im_name, dec)

        print('Saved results for {} into {}...'.format(itr, result_dir))

        self.G.train()

    def interpolate(self, itr):
        batch_size = self.cfg.sample_param.num_samples
        interp_poses = self.cfg.sample_param.interp_poses

        if not self.loaded_interp:
            self.interp_sta, self.interp_end = self.data_loader.dataset.get_interp_imgs(
                interp_poses[0], interp_poses[-1], batch_size, self.device)

            self.interp_sta_T = self.pose_images[interp_poses[0]].unsqueeze(0).repeat(batch_size, 1, 1, 1).to(
                self.device)
            self.interp_end_T = self.pose_images[interp_poses[-1]].unsqueeze(0).repeat(batch_size, 1, 1, 1).to(
                self.device)

            self.loaded_interp = True

        with torch.no_grad():
            self.G.eval()

            max_i = len(interp_poses) - 1
            interp_im = [self.interp_sta]

            for i in range(max_i + 1):
                cur_pose_degree = interp_poses[i]
                cur_pose_T = self.pose_images[cur_pose_degree].unsqueeze(0).repeat(batch_size, 1, 1, 1).to(self.device)
                decoded_image = self.G(self.interp_sta, self.interp_end, self.interp_sta_T, cur_pose_T,
                                       self.interp_end_T)

                interp_im.append(decoded_image)
            interp_im.append(self.interp_end)

        imgs = torch.cat(interp_im, dim=3)

        sample_path = osp.join(self.cfg.dirs.interpolate_dir, '{}-images.jpg'.format(itr + 1))
        save_image(denorm(imgs.data.cpu()), sample_path, nrow=1, padding=0)
        self.G.train()

        print('Saved interpolated images for {} into {}...'.format(itr, sample_path))

    @staticmethod
    def gradient_penalty(y, x, device):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(device)
        dydx = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=weight,
                                   # retain_graph=True,
                                   create_graph=True, only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)
