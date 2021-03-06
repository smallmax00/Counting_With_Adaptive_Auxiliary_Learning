import torch
from torch import nn
from torch.utils import model_zoo
from torch.nn import functional as F
from .GSRU import GCN_Unit_2D


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.vgg = VGG()
        self.load_vgg()
        self.back_end = BackEnd()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.com_layer = nn.Conv2d(64, 32, kernel_size=1)

        self.conv_att = BaseConv(32, 1, 1, 1, use_bn=False)
        self.conv_out = BaseConv(32, 1, 1, 1)
        self.conv_sem = BaseConv(32, 4, 1, 1)

        self.conv_att_central_first = BaseConv(32, 4, 1, 1)
        self.conv_att_central_second = BaseConv(4, 1, 1, 1, activation=nn.ReLU(), use_bn=False)
        self.graphcnn = GCN_Unit_2D(4, 4, normalize=True)

    def _fuse(self, x1, x2):
            x_com = torch.cat((x1, x2), 1)
            x = self.com_layer(x_com)
            return x

    def forward(self, input):
        input = self.vgg(input)
        backend_binary, backend_semantic, backend_density = self.back_end(*input)
        amp_out = self.conv_att(backend_binary)
        smp_out = self.conv_sem(backend_semantic)
        dmp_out_middle = amp_out * backend_density
        dmp_out_late = self.conv_att_central_first(dmp_out_middle)

        dmp_out_late_middle = self.graphcnn(dmp_out_late, smp_out)

        dmp_out_late_middle_later = self.conv_att_central_second(self.upsample(dmp_out_late_middle))
        amp_out = F.sigmoid(self.upsample(amp_out))
        smp_out = self.upsample(smp_out)


        return dmp_out_late_middle_later, amp_out, smp_out



    def load_vgg(self):
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth')
        old_name = [0, 1, 3, 4, 7, 8, 10, 11, 14, 15, 17, 18, 20, 21, 24, 25, 27, 28, 30, 31, 34, 35, 37, 38, 40, 41]
        new_name = ['1_1', '1_2', '2_1', '2_2', '3_1', '3_2', '3_3', '4_1', '4_2', '4_3', '5_1', '5_2', '5_3']
        new_dict = {}
        for i in range(13):
            new_dict['conv' + new_name[i] + '.conv.weight'] = \
                state_dict['features.' + str(old_name[2 * i]) + '.weight']
            new_dict['conv' + new_name[i] + '.conv.bias'] = \
                state_dict['features.' + str(old_name[2 * i]) + '.bias']
            new_dict['conv' + new_name[i] + '.bn.weight'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.weight']
            new_dict['conv' + new_name[i] + '.bn.bias'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.bias']
            new_dict['conv' + new_name[i] + '.bn.running_mean'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.running_mean']
            new_dict['conv' + new_name[i] + '.bn.running_var'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.running_var']

        self.vgg.load_state_dict(new_dict)


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_1 = BaseConv(3, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv1_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_1 = BaseConv(64, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_1 = BaseConv(128, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_3 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_1 = BaseConv(256, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, input):
        input = self.conv1_1(input)
        input = self.conv1_2(input)
        input = self.pool(input)
        input = self.conv2_1(input)
        conv2_2 = self.conv2_2(input)

        input = self.pool(conv2_2)
        conv3_1 = self.conv3_1(input)
        input = self.conv3_2(conv3_1)
        conv3_3 = self.conv3_3(input)

        input = self.pool(conv3_3)
        conv4_1 = self.conv4_1(input)
        input = self.conv4_2(conv4_1)
        conv4_3 = self.conv4_3(input)

        input = self.pool(conv4_3)
        conv5_1 = self.conv5_1(input)
        input = self.conv5_2(conv5_1)
        conv5_3 = self.conv5_3(input)

        return conv2_2, conv3_1, conv3_3, conv4_1, conv4_3, conv5_1, conv5_3



class BackEnd(nn.Module):
    def __init__(self):
        super(BackEnd, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv1 = BaseConv(1024, 256, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv3 = BaseConv(512, 128, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv5 = BaseConv(256, 64, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv6 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv7 = BaseConv(64, 32, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.att_1_first = self.att_layer([512, 512, 512])
        self.att_2_first = self.att_layer([512, 512, 512])
        self.att_3_first = self.att_layer([256, 256, 256])

        self.att_1 = self.att_layer([1024, 512, 512])
        self.att_2 = self.att_layer([1024, 512, 512])
        self.att_3 = self.att_layer([512, 256, 256])

    def att_layer(self, channel):
        att_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[2]),
            nn.Sigmoid(),
        )
        return att_block

    def forward(self, *input):
        conv2_2, conv3_1, conv3_3, conv4_1, conv4_3, conv5_1, conv5_3 = input


        conv5_1_t1 = self.att_1_first(conv5_1)
        conv5_3_t1 = conv5_1_t1 * conv5_3

        conv4_1_t1 = self.att_2_first(conv4_1)
        conv4_3_t1 = conv4_1_t1 * conv4_3

        conv3_1_t1 = self.att_3_first(conv3_1)
        conv3_3_t1 = conv3_1_t1 * conv3_3


        input_t1 = self.upsample(conv5_3_t1)
        input_t1 = torch.cat([input_t1, conv4_3_t1], 1)
        input_t1 = self.conv1(input_t1)
        input_t1 = self.conv2(input_t1)
        input_t1 = self.upsample(input_t1)

        input_t1 = torch.cat([input_t1, conv3_3_t1], 1)
        input_t1 = self.conv3(input_t1)
        input_t1 = self.conv4(input_t1)
        input_t1 = self.upsample(input_t1)

        input_t1 = torch.cat([input_t1, conv2_2], 1)
        input_t1 = self.conv5(input_t1)
        input_t1 = self.conv6(input_t1)
        input_t1 = self.conv7(input_t1)

        merge_conv5_1_2 = torch.cat([conv5_1, conv5_1_t1], 1)
        merge_conv4_1_2 = torch.cat([conv4_1, conv4_1_t1], 1)
        merge_conv3_1_2 = torch.cat([conv3_1, conv3_1_t1], 1)

        conv5_1_t2 = self.att_1(merge_conv5_1_2)
        conv5_3_t2 = conv5_1_t2 * conv5_3

        conv4_1_t2 = self.att_2(merge_conv4_1_2)
        conv4_3_t2 = conv4_1_t2 * conv4_3

        conv3_1_t2 = self.att_3(merge_conv3_1_2)
        conv3_3_t2 = conv3_1_t2 * conv3_3


        input_t2 = self.upsample(conv5_3_t2)
        input_t2 = torch.cat([input_t2, conv4_3_t2], 1)
        input_t2 = self.conv1(input_t2)
        input_t2 = self.conv2(input_t2)
        input_t2 = self.upsample(input_t2)

        input_t2 = torch.cat([input_t2, conv3_3_t2], 1)
        input_t2 = self.conv3(input_t2)
        input_t2 = self.conv4(input_t2)
        input_t2 = self.upsample(input_t2)

        input_t2 = torch.cat([input_t2, conv2_2], 1)
        input_t2 = self.conv5(input_t2)
        input_t2 = self.conv6(input_t2)
        input_t2 = self.conv7(input_t2)

        merge_conv5_1_2_3 = torch.cat([conv5_1, conv5_1_t2], 1)
        merge_conv4_1_2_3 = torch.cat([conv4_1, conv4_1_t2], 1)
        merge_conv3_1_2_3 = torch.cat([conv3_1, conv3_1_t2], 1)

        conv5_1_t3 = self.att_1(merge_conv5_1_2_3)
        conv5_3_t3 = conv5_1_t3 * conv5_3

        conv4_1_t3 = self.att_2(merge_conv4_1_2_3)
        conv4_3_t3 = conv4_1_t3 * conv4_3

        conv3_1_t3 = self.att_3(merge_conv3_1_2_3)
        conv3_3_t3 = conv3_1_t3 * conv3_3


        input_t3 = self.upsample(conv5_3_t3)
        input_t3 = torch.cat([input_t3, conv4_3_t3], 1)
        input_t3 = self.conv1(input_t3)
        input_t3 = self.conv2(input_t3)
        input_t3 = self.upsample(input_t3)

        input_t3 = torch.cat([input_t3, conv3_3_t3], 1)
        input_t3 = self.conv3(input_t3)
        input_t3 = self.conv4(input_t3)
        input_t3 = self.upsample(input_t3)

        input_t3 = torch.cat([input_t3, conv2_2], 1)
        input_t3 = self.conv5(input_t3)
        input_t3 = self.conv6(input_t3)
        input_t3 = self.conv7(input_t3)



        return input_t1, input_t2, input_t3


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, activation=None, use_bn=False):
        super(BaseConv, self).__init__()
        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2)
        self.conv.weight.data.normal_(0, 0.01)
        self.conv.bias.data.zero_()
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.bn(input)
        if self.activation:
            input = self.activation(input)

        return input




