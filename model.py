import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from my_module import ResidualBlock, ConvBlock, TransposeConvBlock, SNConvBlock, SNResidualBlock, SelfAttnBlock
from torch import cat
from solver_utils import vis_tensor


def conv311(in_filters, out_filters):
    """Changes the number of filters"""
    return nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1)


class GenX2(nn.Module):
    """Encodes the template for the target image using a separate and a smaller net"""

    def __init__(self, img_size):
        super().__init__()
        self.filt_const = 1

        # encode input
        self.build_encoder_input()
        self.build_encoder_template()
        self.to_decoder = conv311(self.encoded_input_filts + self.encoded_template_filts, self.encoded_input_filts)

        # decoder
        self.build_decoder()

        # make image
        self.from_decoder = conv311(self.conv_filters, 3)

    def build_encoder_input(self):
        filt_const = self.filt_const
        self.enc_input = nn.Sequential(
            conv311(4, 32 * self.filt_const),
            ConvBlock(32 * filt_const, 64 * filt_const),
            ConvBlock(64 * filt_const, 128 * filt_const),
            SelfAttnBlock(128 * filt_const),
            ConvBlock(128 * filt_const, 256 * filt_const)
        )

        self.encoded_input_filts = 256 * filt_const

    def build_encoder_template(self):
        template_dim = 1
        conv_filts = 4
        encoder_block = [conv311(template_dim, conv_filts)]

        for _ in range(conv_repeat):
            encoder_block.append(ConvBlock(conv_filts, conv_filts))

        self.encode_template = nn.Sequential(*encoder_block)
        self.encoded_template_filts = conv_filts

    def build_decoder(self):
        conv_repeat = 3
        resnet_repeat = 2
        cur_filts = self.encoded_input_filts
        decoder_block = []

        for _ in range(resnet_repeat):
            decoder_block.append(ResidualBlock(cur_filts, cur_filts))

        for _ in range(conv_repeat):
            decoder_block.append(TransposeConvBlock(cur_filts, cur_filts // 2))
            cur_filts = cur_filts // 2

        self.decoder = nn.Sequential(*decoder_block)

    def encode_input_forward(self, im_x, p_x):
        # concact input image and input template
        input_feat = torch.cat([im_x, p_x], dim=1)

        # Encode input feats
        to_encoder = self.to_encoder_input(input_feat)
        encoded_input = self.encode_input(to_encoder)
        return encoded_input

    def decode_forward(self, encoded_input, encoded_template):

        # Concat encoded input feat and target template
        encoded = torch.cat([encoded_input, encoded_template], dim=1)

        # Decode feature maps
        to_decoder = self.to_decoder(encoded)
        im_y_hat = self.from_decoder(self.decoder(to_decoder))

        return im_y_hat

    def forward(self, a, c, a_T, b_T, c_T):

        encoded_input = self.encode_input_forward(a, a_T)

        # Encode target template
        encoded_template = self.encode_template(b_T)

        im_y_hat = self.decode_forward(encoded_input, encoded_template)

        return im_y_hat

    def interpolate(self, a_real, a_T, c_real, c_T, b_T, a_contrib, c_contrib):

        a_encoded = self.encode_input_forward(a_real, a_T)
        c_encoded = self.encode_input_forward(c_real, c_T)

        b_T_encoded = self.encode_template(b_T)
        b_encoded = a_contrib * a_encoded + c_contrib * c_encoded
        b_fake = self.decode_forward(b_encoded, b_T_encoded)

        return b_fake


class GenC(nn.Module):
    def __init__(self, im_size):
        super().__init__()
        self.im_size = im_size
        feed_input = 5

        filt_dims = [32, 64, 128, 256]
        filt_const = 1

        self.to_enc = conv311(feed_input, 32 * filt_const)
        self.enc = nn.Sequential(
            ConvBlock(32 * filt_const, 64 * filt_const),
            ConvBlock(64 * filt_const, 128 * filt_const),
            SelfAttnBlock(128 * filt_const),
            ConvBlock(128 * filt_const, 256 * filt_const)
        )

        self.res = nn.Sequential(ResidualBlock(256 * filt_const, 256 * filt_const),
                                 ResidualBlock(256 * filt_const, 256 * filt_const))

        self.dec = nn.Sequential(
            TransposeConvBlock(256 * filt_const, 128 * filt_const),
            SelfAttnBlock(128 * filt_const),
            TransposeConvBlock(128 * filt_const, 64 * filt_const),
            # SelfAttnBlock(64 * filt_const)
            TransposeConvBlock(64 * filt_const, 32 * filt_const),
        )

        self.from_dec = conv311(32 * filt_const, 3)

    def forward(self, a, c, a_T, b_T, c_T):
        # a + c --> b
        # x = cat([a, c, a_T, c_T, b_T], dim=1)
        # x = cat([a, c, a_T, b_T, c_T], dim=1)
        x = cat([a, a_T, b_T], dim=1)

        x = self.to_enc(x)
        x = self.enc(x)
        x = self.res(x)
        x = self.dec(x)
        c_hat = self.from_dec(x)

        return c_hat


class GenB(nn.Module):
    def __init__(self, im_size):
        super().__init__()
        self.im_size = im_size
        # feed to encoder

        # feed_input = 7
        # feed_input = 9
        feed_input = 5

        filt_dims = [32, 64, 128, 256]
        filt_const = 1
        self.to_enc = conv311(feed_input, 32 * filt_const)
        self.e1 = ConvBlock(32 * filt_const, 64 * filt_const)
        self.e2 = ConvBlock(64 * filt_const, 128 * filt_const)
        self.e2_e3 = SelfAttnBlock(128 * filt_const)
        self.e3 = ConvBlock(128 * filt_const, 256 * filt_const)

        self.res = nn.Sequential(ResidualBlock(256 * filt_const, 256 * filt_const),
                                 ResidualBlock(256 * filt_const, 256 * filt_const))

        self.d3 = TransposeConvBlock(256 * filt_const, 128 * filt_const)
        self.d3_d2 = SelfAttnBlock(128 * filt_const)
        self.d2 = TransposeConvBlock(128 * filt_const, 64 * filt_const)
        self.d2_d1 = SelfAttnBlock(64 * filt_const)
        self.d1 = TransposeConvBlock(64 * filt_const, 32 * filt_const)

        self.from_dec = conv311(32 * filt_const, 3)

    def forward(self, a, c, a_T, b_T, c_T):
        # a + c --> b
        # x = cat([a, c, a_T, c_T, b_T], dim=1)
        # x = cat([a, c, a_T, b_T, c_T], dim=1)
        x = cat([a, a_T, b_T], dim=1)

        x = self.to_enc(x)
        e1_out = self.e1(x)
        e2_out = self.e2(e1_out)
        e2_e3_out = self.e2_e3(e2_out)

        e3_out = self.e3(e2_e3_out)
        res_out = self.res(e3_out)
        d2_in = self.d3(res_out)

        d3_d2_in = self.d3_d2(d2_in)
        d1_in = self.d2(d3_d2_in)
        d1_in = self.d2_d1(d1_in)

        x = self.d1(d1_in)
        c_hat = self.from_dec(x)
        return c_hat


class GenA(nn.Module):
    def __init__(self, im_size):
        super().__init__()
        self.im_size = im_size
        # feed to encoder

        # feed_input = 7
        # feed_input = 9
        feed_input = 5

        filt_dims = [32, 64, 128, 256]
        filt_const = 1
        self.to_enc = conv311(feed_input, 32 * filt_const)
        self.e1 = ConvBlock(32 * filt_const, 64 * filt_const)
        self.e2 = ConvBlock(64 * filt_const, 128 * filt_const)
        self.e2_e3 = SelfAttnBlock(128 * filt_const)
        self.e3 = ConvBlock(128 * filt_const, 256 * filt_const)

        self.res = nn.Sequential(ResidualBlock(256 * filt_const, 256 * filt_const),
                                 ResidualBlock(256 * filt_const, 256 * filt_const))

        self.d3 = TransposeConvBlock(256 * filt_const, 128 * filt_const)
        self.d3_d2 = SelfAttnBlock(128 * filt_const)
        self.d2 = TransposeConvBlock(256 * filt_const, 64 * filt_const)
        self.d1 = TransposeConvBlock(64 * filt_const, 32 * filt_const)

        self.from_dec = conv311(32 * filt_const, 3)

    def forward(self, a, c, a_T, b_T, c_T):
        # a + c --> b
        # x = cat([a, c, a_T, c_T, b_T], dim=1)
        x = cat([a, a_T, b_T], dim=1)
        # x = cat([a, a_T, b_T], dim=1)

        x = self.to_enc(x)
        e1_out = self.e1(x)
        e2_out = self.e2(e1_out)

        e2_e3_out = self.e2_e3(e2_out)

        e3_out = self.e3(e2_e3_out)
        res_out = self.res(e3_out)
        d2_in = self.d3(res_out)

        d3_d2_in = self.d3_d2(d2_in)

        d1_in = self.d2(cat([d3_d2_in, e2_out], dim=1))
        x = self.d1(d1_in)

        c_hat = self.from_dec(x)

        return c_hat


class GenZ(nn.Module):
    def __init__(self, dev_id):
        super().__init__()
        self.dev_id = dev_id
        self.noise_dim = 50
        feed_input = 5

        filt_dims = [32, 64, 128, 256]
        filt_const = 2
        self.to_enc = conv311(feed_input, 32 * filt_const)
        self.e1 = ConvBlock(32 * filt_const, 64 * filt_const)
        self.e2 = ConvBlock(64 * filt_const, 128 * filt_const)
        self.e3 = ConvBlock(128 * filt_const, 256 * filt_const)
        # self.e3 = ConvBlock(128, 128) for attention change to 128

        self.res = nn.Sequential(ResidualBlock(256 * filt_const, 256 * filt_const),
                                 ResidualBlock(256 * filt_const, 256 * filt_const))

        self.d3 = TransposeConvBlock(256 * filt_const + self.noise_dim // 2, 128 * filt_const)
        self.d2 = TransposeConvBlock(256 * filt_const, 64 * filt_const)
        self.d1 = TransposeConvBlock(128 * filt_const, 32 * filt_const)

        self.from_dec = conv311(32 * filt_const, 3)

        self.noise_feat = nn.Sequential(
            TransposeConvBlock(self.noise_dim, self.noise_dim // 2, k=2, s=1, p=1),
            TransposeConvBlock(self.noise_dim // 2, self.noise_dim // 2, k=2, ),
            TransposeConvBlock(self.noise_dim // 2, self.noise_dim // 2),
            TransposeConvBlock(self.noise_dim // 2, self.noise_dim // 2))

    #
    # def get_noise(self):
    #     return torch.randn(self.noise_dim).to(self.dev_id)

    def forward(self, a, c, a_T, b_T, c_T):
        # a + c --> b
        # x = cat([a, c, a_T, c_T, b_T], dim=1)
        # x = cat([a, c, a_T, b_T, c_T], dim=1)
        x = cat([a, a_T, b_T], dim=1)

        x = self.to_enc(x)
        e1_out = self.e1(x)
        e2_out = self.e2(e1_out)
        e3_out = self.e3(e2_out)

        res_out = self.res(e3_out)
        # noise_feat = self.noise_feat(self.get_noise())

        d2_in = self.d3(cat(res_out, noise_feat))
        d1_in = self.d2(cat([d2_in, e2_out], dim=1))
        x = self.d1(cat([d1_in, e1_out], dim=1))

        c_hat = self.from_dec(x)

        return c_hat


class SNDisA(nn.Module):
    def __init__(self, im_size, num_classes):
        super().__init__()
        feed_input = 4

        # a b_real; a b_fake -> output weather b_real corresponds to a or not
        # b_T b_real; b_t b_fake -> output weather b_t and b_real are correlated and also extract image id from image features

        # extract_img_features -> works on all images ->

        q = int(np.log2(im_size))
        filt_dims = [2 ** (12 - i) for i in range(q + 1, 1, -1)]

        down_sample = 4
        block = [conv311(3, filt_dims[0])]
        for i in range(len(filt_dims) - 1):
            block.append(SNConvBlock(filt_dims[i], filt_dims[i + 1]))
            if i != len(filt_dims) - 2:
                block.append(SNResidualBlock(filt_dims[i + 1], filt_dims[i + 1]))
        self.from_rgb = nn.Sequential(*block)

        block = [conv311(1, filt_dims[0])]
        for i in range(len(filt_dims) - 1):
            block.append(SNConvBlock(filt_dims[0], filt_dims[0]))
        self.from_template = nn.Sequential(*block)

        self.conv_real = nn.Conv2d(filt_dims[-1], 1, kernel_size=2, stride=1, padding=0)
        self.conv_class = nn.Conv2d(filt_dims[-1], num_classes, kernel_size=2, stride=1, padding=0)
        self.num_classes = num_classes
        # self.filter_list = filt_dims

    def forward(self, a, b, c, a_T, b_T, c_T):
        batch_size = a.shape[0]

        # x = cat([a, b, c, a_T, b_T, c_T], dim=1)
        x = cat([b, b_T], dim=1)

        x = self.from_rgb(x)
        out = self.down_sample(x)

        out_src = self.conv_real(out)
        out_src = out_src.view(batch_size, 1)

        out_class = self.conv_class(out)
        out_class = out_class.view(batch_size, self.num_classes)

        return out_src, out_class


class SNDisZ(nn.Module):
    def __init__(self, im_size, num_classes):
        super().__init__()
        # feed_input = 10
        feed_input = 4

        q = int(np.log2(im_size))  # 7
        filt_dims = [2 ** (12 - i) for i in range(q, 1, -1)]
        print("Discriminator Filters: ", filt_dims)

        self.from_rgb = conv311(feed_input, filt_dims[0])

        block = []
        for i in range(len(filt_dims) - 1):
            block.append(SNConvBlock(filt_dims[i], filt_dims[i + 1]))
            if i != len(filt_dims) - 2:
                block.append(SNResidualBlock(filt_dims[i + 1], filt_dims[i + 1]))
        block.append(SNConvBlock(filt_dims[-1], filt_dims[-1]))

        self.down_sample = nn.Sequential(*block)

        self.conv_real = nn.Conv2d(filt_dims[-1], 1, kernel_size=2, stride=1, padding=0)
        self.conv_class = nn.Conv2d(filt_dims[-1], num_classes, kernel_size=2, stride=1, padding=0)
        self.num_classes = num_classes
        # self.filter_list = filt_dims

    def forward(self, a, b, c, a_T, b_T, c_T):
        batch_size = a.shape[0]

        # x = cat([a, b, c, a_T, b_T, c_T], dim=1)
        x = cat([b, b_T], dim=1)

        x = self.from_rgb(x)
        out = self.down_sample(x)

        out_src = self.conv_real(out)
        out_src = out_src.view(batch_size, 1)

        out_class = self.conv_class(out)
        out_class = out_class.view(batch_size, self.num_classes)

        return out_src, out_class


class DisZ(nn.Module):
    def __init__(self, im_size, num_classes):
        super().__init__()
        # feed_input = 10
        feed_input = 8

        q = int(np.log2(im_size))
        filt_dims = [2 ** (12 - i) for i in range(q, 1, -1)]
        print("Discriminator Filters: ", filt_dims)

        self.from_rgb = conv311(feed_input, filt_dims[0])

        block = []
        for i in range(len(filt_dims) - 1):
            block.append(ConvBlock(filt_dims[i], filt_dims[i + 1]))
            if i != len(filt_dims) - 2:
                block.append(ResidualBlock(filt_dims[i + 1], filt_dims[i + 1]))
        block.append(ConvBlock(filt_dims[-1], filt_dims[-1]))

        self.down_sample = nn.Sequential(*block)

        self.conv_real = nn.Conv2d(filt_dims[-1], 1, kernel_size=2, stride=1, padding=0)
        self.conv_class = nn.Conv2d(filt_dims[-1], num_classes, kernel_size=2, stride=1, padding=0)
        self.num_classes = num_classes
        # self.filter_list = filt_dims

    def forward(self, a, b, c, a_T, b_T, c_T):
        batch_size = a.shape[0]

        # x = cat([a, b, c, a_T, b_T, c_T], dim=1)
        x = cat([a, b, a_T, b_T], dim=1)

        x = self.from_rgb(x)
        out = self.down_sample(x)

        out_src = self.conv_real(out)
        out_src = out_src.view(batch_size, 1)

        out_class = self.conv_class(out)
        out_class = out_class.view(batch_size, self.num_classes)

        return out_src, out_class


class GenY(nn.Module):
    """Generates feature maps for target pose"""

    def __init__(self, im_size):
        super().__init__()
        self.im_size = im_size
        # feed to encoder
        image_dim = 3
        template_dim = 1
        input_dim = image_dim + template_dim

        # can change?
        self.filt_dims = [32, 64, 128, 256]

        self.to_enc = conv311(input_dim, self.filt_dims[0])
        self.make_encoder()
        self.make_decoder()

        self.to_decoder = conv311(self.encoded_input_filts + self.encoded_template_filts, self.encoded_input_filts)

        # decoder
        self.build_decoder()

        # make image
        self.from_decoder = conv311(self.conv_filters, 3)

    def make_encoder(self):
        filt_dims = self.filt_dims

        enc = nn.ModuleList()
        for i_dim in range(len(filt_dims) - 1):
            enc.append([ConvBlock(filt_dims[i_dim], filt_dims[i_dim + 1])])

        self.enc = enc
        self.enc_res = ResidualBlock(filt_dims[-1], filt_dims[-1])

        self.enc_im_size = self.im_size // 2 ** len(enc)

    def make_decoder(self):
        filt_dims = self.filt_dims[::-1]
        self.dec_res = ResidualBlock(2 * filt_dims[-1], filt_dims[-1])

        dec = nn.ModuleList()
        for i_dim in range(len(filt_dims) - 1):
            dec.append([TransposeConvBlock(filt_dims[i_dim], TransposeConvBlock[i_dim + 1])])

        self.dec = dec
        self.dec_im_size = self.im_size * 2 ** len(dec)

    def make_feat_pose(self):
        last_filt_dim = self.filt_dims[-1]

        # im_size = 1
        self.feat_temp = nn.Sequential(TransposeConvBlock(1, 32),  # im_size = 2
                                       TransposeConvBlock(32, 32),  # im_size = 4
                                       TransposeConvBlock(32, 32),  # im_size = 8
                                       TransposeConvBlock(32, last_filt_dim))  # im_size = 16

    def encode_input_forward(self, im_x, p_x):
        # concact input image and input template
        input_feat = cat([im_x, p_x], dim=1)

        # Encode input feats
        to_encoder = self.to_enc(input_feat)
        encoded_input = self.encode_input(to_encoder)
        return encoded_input

    def decode_forward(self, encoded_input, encoded_template):

        # Concat encoded input feat and target template
        encoded = torch.cat([encoded_input, encoded_template], dim=1)

        # Decode feature maps
        to_decoder = self.to_decoder(encoded)
        im_y_hat = self.from_decoder(self.decoder(to_decoder))

        return im_y_hat

    def forward(self, im_x, p_x, p_y, ):

        encoded_input = self.encode_input_forward(im_x, p_x)

        # Encode target template
        encoded_template = self.encode_template(p_y)

        im_y_hat = self.decode_forward(encoded_input, encoded_template)

        return im_y_hat

    def interpolate(self, a_real, a_T, c_real, c_T, b_T, a_contrib, c_contrib):

        a_encoded = self.encode_input_forward(a_real, a_T)
        c_encoded = self.encode_input_forward(c_real, c_T)

        b_T_encoded = self.encode_template(b_T)
        b_encoded = a_contrib * a_encoded + c_contrib * c_encoded
        b_fake = self.decode_forward(b_encoded, b_T_encoded)

        return b_fake


class GenX(nn.Module):
    """Encodes the template for the target image using a separate and a smaller net"""

    def __init__(self, img_size):
        super().__init__()
        # imsize = 128
        self.conv_repeat = 3
        self.resnet_repeat = 2
        self.conv_filters = 32

        # feed to encoder
        image_dim = 3
        template_dim = 1
        input_dim = image_dim + template_dim
        self.to_encoder_input = conv311(input_dim, self.conv_filters)

        # encode input
        self.build_encoder_input()

        # encode template
        self.build_encoder_template()

        self.to_decoder = conv311(self.encoded_input_filts + self.encoded_template_filts, self.encoded_input_filts)

        # decoder
        self.build_decoder()

        # make image
        self.from_decoder = conv311(self.conv_filters, 3)

    def build_encoder_input(self):
        conv_filters = self.conv_filters
        conv_repeat = self.conv_repeat
        resnet_repeat = self.resnet_repeat

        encoder_block = []
        cur_filts = conv_filters
        for _ in range(conv_repeat):
            encoder_block.append(ConvBlock(cur_filts, cur_filts * 2))
            cur_filts = cur_filts * 2

        for _ in range(resnet_repeat):
            encoder_block.append(ResidualBlock(cur_filts, cur_filts))

        self.encoded_input_filts = cur_filts
        self.encode_input = nn.Sequential(*encoder_block)

    def build_encoder_template(self):
        template_dim = 1
        conv_repeat = self.conv_repeat
        conv_filts = 4
        encoder_block = [conv311(template_dim, conv_filts)]

        for _ in range(conv_repeat):
            encoder_block.append(ConvBlock(conv_filts, conv_filts))

        self.encode_template = nn.Sequential(*encoder_block)
        self.encoded_template_filts = conv_filts

    def build_decoder(self):
        conv_repeat = 3
        resnet_repeat = 2
        cur_filts = self.encoded_input_filts
        decoder_block = []

        for _ in range(resnet_repeat):
            decoder_block.append(ResidualBlock(cur_filts, cur_filts))

        for _ in range(conv_repeat):
            decoder_block.append(TransposeConvBlock(cur_filts, cur_filts // 2))
            cur_filts = cur_filts // 2

        self.decoder = nn.Sequential(*decoder_block)

    def encode_input_forward(self, im_x, p_x):
        # concact input image and input template
        input_feat = torch.cat([im_x, p_x], dim=1)

        # Encode input feats
        to_encoder = self.to_encoder_input(input_feat)
        encoded_input = self.encode_input(to_encoder)
        return encoded_input

    def decode_forward(self, encoded_input, encoded_template):

        # Concat encoded input feat and target template
        encoded = torch.cat([encoded_input, encoded_template], dim=1)

        # Decode feature maps
        to_decoder = self.to_decoder(encoded)
        im_y_hat = self.from_decoder(self.decoder(to_decoder))

        return im_y_hat

    def forward(self, a, c, a_T, b_T, c_T):

        encoded_input = self.encode_input_forward(a, a_T)

        # Encode target template
        encoded_template = self.encode_template(b_T)

        im_y_hat = self.decode_forward(encoded_input, encoded_template)

        return im_y_hat

    def interpolate(self, a_real, a_T, c_real, c_T, b_T, a_contrib, c_contrib):

        a_encoded = self.encode_input_forward(a_real, a_T)
        c_encoded = self.encode_input_forward(c_real, c_T)

        b_T_encoded = self.encode_template(b_T)
        b_encoded = a_contrib * a_encoded + c_contrib * c_encoded
        b_fake = self.decode_forward(b_encoded, b_T_encoded)

        return b_fake


class GeneratorUNet(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        feed_input = 5 + 4
        conv_dim = 32

        self.layer_1 = nn.Sequential(nn.Conv2d(feed_input, conv_dim, kernel_size=3, stride=1, padding=1),
                                     ConvBlock(conv_dim, conv_dim, k=3, s=1))

        self.layer_2 = nn.Sequential(ConvBlock(conv_dim, conv_dim * 2),
                                     ConvBlock(conv_dim * 2, conv_dim * 2, k=3, s=1))

        self.layer_3 = nn.Sequential(ConvBlock(conv_dim * 2, conv_dim * 4),
                                     ConvBlock(conv_dim * 4, conv_dim * 4, k=3, s=1))

        self.layer_4 = nn.Sequential(ConvBlock(conv_dim * 4, conv_dim * 8),
                                     ConvBlock(conv_dim * 8, conv_dim * 8, k=3, s=1))

        self.bottom_upsample = nn.Sequential(ConvBlock(conv_dim * 8, conv_dim * 16),
                                             ConvBlock(conv_dim * 16, conv_dim * 32, k=3, s=1, p=1),
                                             TransposeConvBlock(conv_dim * 32, conv_dim * 8))  # 8 - > 16

        self.upsampling_1 = nn.Sequential(ConvBlock(conv_dim * 16, conv_dim * 8, k=3, s=1, p=1),
                                          ConvBlock(conv_dim * 8, conv_dim * 8, k=3, s=1),
                                          TransposeConvBlock(conv_dim * 8, conv_dim * 4))  # 16 -> 32

        self.upsampling_2 = nn.Sequential(ConvBlock(conv_dim * 8, conv_dim * 4, k=3, s=1, p=1),
                                          ConvBlock(conv_dim * 4, conv_dim * 4, k=3, s=1),
                                          TransposeConvBlock(conv_dim * 4, conv_dim * 2))  # 32 -> 64

        self.upsampling_3 = nn.Sequential(ConvBlock(conv_dim * 4, conv_dim * 2, k=3, s=1, p=1),
                                          ConvBlock(conv_dim * 2, conv_dim * 2, k=3, s=1),
                                          TransposeConvBlock(conv_dim * 2, conv_dim))  # 64 -> 128

        self.final = nn.Sequential(ConvBlock(conv_dim * 2, conv_dim, k=3, s=1, p=1),
                                   nn.Conv2d(conv_dim, 3, stride=1, padding=1, kernel_size=3))

    def forward(self, a, c, a_T, b_T, c_T):
        # a + c --> b
        x = torch.cat([a, a_T, b_T, c, c_T], dim=1)

        h_1 = self.layer_1(x)  # 128 -> 128
        h_2 = self.layer_2(h_1)  # 128 -> 64
        h_3 = self.layer_3(h_2)  # 64 -> 32
        h_4 = self.layer_4(h_3)  # 32 -> 16

        up_1 = self.bottom_upsample(h_4)  # 16 -> 8 -> 16
        up_2 = self.upsampling_1(torch.cat([up_1, h_4], dim=1))  # 16 -> 32
        up_3 = self.upsampling_2(torch.cat([up_2, h_3], dim=1))  # 32 -> 64
        up_4 = self.upsampling_3(torch.cat([up_3, h_2], dim=1))  # 64 -> 128
        out = self.final(torch.cat([h_1, up_4], dim=1))  # 128 -> 128

        return out
