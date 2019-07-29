import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO : change initialisation to Glorot, right now using pytorch 1.0 to do that same


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization. Doesn't downsize the input feature maps"""

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.05, inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class ConvBlockDropout(nn.Module):
    """Downsizes the input feature map by a factor of 2"""

    def __init__(self, in_channel, out_channel, k=4, s=2, p=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=k, stride=s, padding=p, bias=False),
            nn.InstanceNorm2d(out_channel, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5))

    def forward(self, x):
        return self.block(x)


class ConvBlock(nn.Module):
    """By Default downsizes the input feature map by a factor of 2"""

    def __init__(self, in_channel, out_channel, k=4, s=2, p=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=k, stride=s, padding=p, bias=False),
            nn.LeakyReLU(0.05, inplace=True),
            nn.InstanceNorm2d(out_channel, affine=True, track_running_stats=True))

    def forward(self, x):
        return self.block(x)


class TransposeConvBlock(nn.Module):
    """By Default upsamples the input feature map by a factor of 2"""

    def __init__(self, in_channel, out_channel, k=4, s=2, p=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=k, stride=s, padding=p, bias=False),
            nn.LeakyReLU(0.05, inplace=True),
            nn.InstanceNorm2d(out_channel, affine=True, track_running_stats=True))

    def forward(self, x):
        return self.block(x)


#
# class SNConvBlock(nn.Module):
#     def __init__(self, in_channel, out_channel, k=4, s=2, p=1):
#         super(SNConvBlock, self).__init__()
#
#
#         self.c = nn.Conv2d(in_channel, out_channel, kernel_size=k, stride=s, padding=p)
#         self.act = nn.LeakyReLU(0.05, inplace=True)
#         torch.nn.utils.spectral_norm(self.c)
#
#     def forward(self, x):
#         return self.act(self.c(x))

def spec_conv2d(in_channel, out_channel, k, s, p):
    return torch.nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=k, stride=s, padding=p))


class SNConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, k=4, s=2, p=1):
        super().__init__()

        self.block = nn.Sequential(
            spec_conv2d(in_channel, out_channel, k, s, p),
            nn.LeakyReLU(0.05, inplace=True))

    def forward(self, x):
        return self.block(x)


class SNResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.main = nn.Sequential(
            spec_conv2d(dim_in, dim_out, 3, 1, 1),
            nn.LeakyReLU(0.05, inplace=True),
            spec_conv2d(dim_in, dim_out, 3, 1, 1))

    def forward(self, x):
        return x + self.main(x)


class SelfAttnBlock(nn.Module):
    def __init__(self, dim_in):
        super().__init__()

        self.main = nn.Sequential(
            SelfAttnConv(dim_in),
            nn.LeakyReLU(0.05, inplace=True),
            nn.InstanceNorm2d(dim_in, affine=True, track_running_stats=True))

    def forward(self, x):
        return self.main(x)


class SelfAttnConv(nn.Module):
    """ Referenced from https://github.com/heykeetae/Self-Attention-GAN"""

    def __init__(self, in_dim):
        super().__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        # return out, attention

        return out

# def weights_init(m):
#     if isinstance(m, nn.Conv2d):
#         torch.nn.init.xavier_uniform_(m.weight)
#         if m.bias:
#             torch.nn.init.xavier_uniform_(m.bias)
