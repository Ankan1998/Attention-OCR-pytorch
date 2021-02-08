import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.inception import BasicConv2d, InceptionA
import scipy.stats as stats


class BasicConv2d_inc_res_2(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d_inc_res_2, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001, # value found in tensorflow
                                 momentum=0.1, # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_5b_inc_res_2(nn.Module):

    def __init__(self):
        super(Mixed_5b_inc_res_2, self).__init__()

        self.branch0 = BasicConv2d_inc_res_2(192, 96, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d_inc_res_2(192, 48, kernel_size=1, stride=1),
            BasicConv2d_inc_res_2(48, 64, kernel_size=5, stride=1, padding=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d_inc_res_2(192, 64, kernel_size=1, stride=1),
            BasicConv2d_inc_res_2(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2d_inc_res_2(96, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d_inc_res_2(192, 64, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super(Block35, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d_inc_res_2(320, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d_inc_res_2(320, 32, kernel_size=1, stride=1),
            BasicConv2d_inc_res_2(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d_inc_res_2(320, 32, kernel_size=1, stride=1),
            BasicConv2d_inc_res_2(32, 48, kernel_size=3, stride=1, padding=1),
            BasicConv2d_inc_res_2(48, 64, kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_6a_inc_res_2(nn.Module):

    def __init__(self):
        super(Mixed_6a_inc_res_2, self).__init__()

        self.branch0 = BasicConv2d_inc_res_2(320, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d_inc_res_2(320, 256, kernel_size=1, stride=1),
            BasicConv2d_inc_res_2(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d_inc_res_2(256, 384, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out



class InceptionResNetV2(nn.Module):

    def __init__(self):
        super(InceptionResNetV2, self).__init__()
        # Modules
        self.conv2d_1a = BasicConv2d_inc_res_2(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d_inc_res_2(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d_inc_res_2(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d_inc_res_2(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d_inc_res_2(80, 192, kernel_size=3, stride=1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = Mixed_5b_inc_res_2()
        self.repeat = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17)
        )
        self.mixed_6a = Mixed_6a_inc_res_2()


    def forward(self, input):
        x = self.conv2d_1a(input)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x = self.mixed_6a(x)

        return x

class MyIncept(nn.Module):
    def __init__(self):
        super(MyIncept, self).__init__()
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):

                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.numel()))
                values = values.view(m.weight.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)

        return x


class OneHot(nn.Module):
    def __init__(self, depth):
        super(OneHot, self).__init__()
        emb = nn.Embedding(depth, depth)
        emb.weight.data = torch.eye(depth)
        emb.weight.requires_grad = False
        self.emb = emb

    def forward(self, input_):
        return self.emb(input_)


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size), requires_grad=True)
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(1)
        h = hidden.expand(timestep, -1, -1).transpose(0, 1)
        attn_energies = self.score(h, encoder_outputs)
        return attn_energies.softmax(2)

    def score(self, hidden, encoder_outputs):
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)
        v = self.v.expand(encoder_outputs.size(0), -1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        return energy


class Decoder(nn.Module):
    def __init__(self, vocab_size, max_len, hidden_size, sos_id, eos_id, n_layers=1):
        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.n_layers = n_layers

        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.attention = Attention(hidden_size)
        self.rnn = nn.GRU(hidden_size * 2, hidden_size, n_layers)

        self.out = nn.Linear(hidden_size, vocab_size)

    def forward_step(self, input_, last_hidden, encoder_outputs):
        emb = self.emb(input_.transpose(0, 1))
        attn = self.attention(last_hidden, encoder_outputs)
        context = attn.bmm(encoder_outputs).transpose(0, 1)
        rnn_input = torch.cat((emb, context), dim=2)

        outputs, hidden = self.rnn(rnn_input, last_hidden)

        if outputs.requires_grad:
            outputs.register_hook(lambda x: x.clamp(min=-10, max=10))

        outputs = self.out(outputs.contiguous().squeeze(0)).log_softmax(1)

        return outputs, hidden

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None,teacher_forcing_ratio=0):
        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs, teacher_forcing_ratio)

        use_teacher_forcing = True if torch.rand(1).item() < teacher_forcing_ratio else False

        outputs = []

        self.rnn.flatten_parameters()

        decoder_hidden = torch.zeros(1, batch_size, self.hidden_size, device=encoder_outputs.device)

        def decode(step_output):
            symbols = step_output.topk(1)[1]
            return symbols

        if use_teacher_forcing:
            for di in range(max_length):
                decoder_input = inputs[:, di].unsqueeze(1)

                decoder_output, decoder_hidden = self.forward_step(
                    decoder_input, decoder_hidden, encoder_outputs)

                step_output = decoder_output.squeeze(1)
                outputs.append(step_output)
        else:
            decoder_input = inputs[:, 0].unsqueeze(1)
            for di in range(max_length):
                decoder_output, decoder_hidden = self.forward_step(
                    decoder_input, decoder_hidden, encoder_outputs
                )

                step_output = decoder_output.squeeze(1)
                outputs.append(step_output)

                symbols = decode(step_output)
                decoder_input = symbols

        outputs = torch.stack(outputs).permute(1, 0, 2)

        return outputs, decoder_hidden

    def _validate_args(self, inputs,encoder_hidden, encoder_outputs, teacher_forcing_ratio):
        batch_size = encoder_outputs.size(0)

        if inputs is None:
            assert teacher_forcing_ratio == 0

            inputs = torch.full((batch_size, 1), self.sos_id, dtype=torch.long, device=encoder_outputs.device)

            max_length = self.max_len
        else:
            max_length = inputs.size(1) - 1

        return inputs, batch_size, max_length


class OCR(nn.Module):
    def __init__(self, img_width, img_height, nh, n_classes, max_len, SOS_token, EOS_token,model_name):
        super(OCR, self).__init__()
        self.model_name=model_name

        
        if self.model_name == "inceptionv3":
            self.channel_change = 288
            self.model_arch = MyIncept()
        else:
            self.channel_change = 1088
            self.model_arch = InceptionResNetV2()
    
        f = self.model_arch(torch.rand(1, 3, img_height, img_width))

        self._fh = f.size(2)
        self._fw = f.size(3)
        print('Model feature size:', self._fh, self._fw)

        self.onehot_x = OneHot(self._fh)
        self.onehot_y = OneHot(self._fw)
        self.encode_emb = nn.Linear(self.channel_change + self._fh + self._fw, nh)
        self.decoder = Decoder(n_classes, max_len, nh, SOS_token, EOS_token)

        # else:
        #
        #     self.model_arch = InceptionResNetV2()
        #
        #     f = self.model_arch(torch.rand(1, 3, img_height, img_width))
        #
        #     self._fh = f.size(2)
        #     self._fw = f.size(3)
        #     print('Model feature size:', self._fh, self._fw)
        #
        #     self.onehot_x = OneHot(self._fh)
        #     self.onehot_y = OneHot(self._fw)
        #     self.encode_emb = nn.Linear(1088 + self._fh + self._fw, nh)
        #     self.decoder = Decoder(n_classes, max_len, nh, SOS_token, EOS_token)
            

    def forward(self, input_, target_seq=None, teacher_forcing_ratio=0):
        device = input_.device
        b, c, h, w = input_.size()

        encoder_outputs = self.model_arch(input_)

        b, fc, fh, fw = encoder_outputs.size()

        # made co-ordinate system
        x, y = torch.meshgrid(torch.arange(fh, device=device), torch.arange(fw, device=device))

        h_loc = self.onehot_x(x)
        w_loc = self.onehot_y(y)

        loc = torch.cat([h_loc, w_loc], dim=2).unsqueeze(0).expand(b, -1, -1, -1)

        encoder_outputs = torch.cat([encoder_outputs.permute(0, 2, 3, 1), loc], dim=3)
        encoder_outputs = encoder_outputs.contiguous().view(b, -1, self.channel_change + self._fh + self._fw)

        encoder_outputs = self.encode_emb(encoder_outputs)

        decoder_outputs, decoder_hidden = self.decoder(target_seq, encoder_outputs=encoder_outputs,
                                                       teacher_forcing_ratio=teacher_forcing_ratio)

        return decoder_outputs



if __name__=="__main__":
    ocr_model=OCR()