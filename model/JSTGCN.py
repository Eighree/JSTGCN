import torch
import torch.nn as nn
import torch.nn.functional as F


class JGCM(nn.Module):
    def __init__(self, step_k, dim_in, dim_out, support_len=1, order=2):
        super(JGCM, self).__init__()
        dim_in = (order * support_len + 1) * dim_in
        self.step_k = step_k
        self.order = order
        self.mlp = torch.nn.Conv2d(dim_in, dim_out, kernel_size=(1, 1),
                                   padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x, sp_matrix, seq_matrix):

        seq = F.normalize(seq_matrix, p=1, dim=1)
        x = torch.einsum('tk,bink->bint', torch.matrix_power(seq, self.step_k), x)
        out = [x]
        for sp in sp_matrix:
            x1 = torch.einsum('ncvl,vw->ncwl', (x, sp))
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = torch.einsum('ncvl,vw->ncwl', (x1, sp))
                out.append(x2)
                x1 = x2

        x_st = torch.cat(out, dim=1)
        x_st = self.mlp(x_st)
        return x_st





class JSTGCN(nn.Module):
    def __init__(self, args):
        super(JSTGCN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.residual_dim = args.rnn_units
        self.dilation_dim = args.rnn_units
        self.skip_dim = args.rnn_units * 8
        self.end_dim = args.rnn_units * 16
        self.input_window = args.horizon
        self.output_window = args.horizon
        self.device = torch.device('cuda:0')

        self.layers = args.num_layers
        self.blocks = args.blocks

        seq_matrix_1 = torch.eye(self.input_window).to(self.device)
        seq_matrix_2 = torch.eye(self.input_window - 1).to(self.device)
        padding = nn.ConstantPad2d((0, 1, 1, 0), 0)
        self.seq_matrix = seq_matrix_1 + padding(seq_matrix_2)

        self.node_embedding = nn.Parameter(torch.randn(3, self.num_node, 10).to(self.device),
                                           requires_grad=True).to(self.device)
        self.weight = nn.Parameter(torch.randn(self.num_node, 3).to(self.device),
                                   requires_grad=True).to(self.device)
        self.start_conv = nn.Conv2d(in_channels=self.input_dim,
                                    out_channels=self.residual_dim,
                                    kernel_size=(1, 1))
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()

        for b in range(self.blocks):
            for i in range(self.layers):
                self.filter_convs.append(JGCM(3**i, self.residual_dim, self.dilation_dim))
                self.gate_convs.append(JGCM(3**i, self.residual_dim, self.dilation_dim))
                self.skip_convs.append(nn.Conv2d(in_channels=self.dilation_dim,
                                                 out_channels=self.skip_dim,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(self.residual_dim))

        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_dim,
                                    out_channels=self.end_dim,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=self.end_dim,
                                    out_channels=self.output_window,
                                    kernel_size=(1, 1),
                                    bias=True)

    def forward(self, source, targets, teacher_forcing_ratio=0.5):
        inputs = source
        inputs = inputs.permute(0, 3, 2, 1)
        x = self.start_conv(inputs)
        skip = 0

        weight = F.softmax(F.relu(self.weight), dim=-1)
        embedding_matrices = torch.einsum('abc,acf->abf', self.node_embedding, self.node_embedding.transpose(1, 2))
        embedding_matrices = F.softmax(F.relu(embedding_matrices), dim=-1)
        embedding_matrices = embedding_matrices.transpose(0, 1)
        sp_matrix = [torch.einsum('nd,ndm->nm', weight, embedding_matrices)]
        seq_matrix = self.seq_matrix

        for i in range(self.blocks * self.layers):
            residual = x   # residual_dim
            filter = self.filter_convs[i](residual, sp_matrix, seq_matrix)   # dilation_dim
            gate = torch.sigmoid(self.gate_convs[i](residual, sp_matrix, seq_matrix))   # dilation_dim
            x = filter * gate  # dilation_dim

            s = x[:, :, :, -1:]
            s = self.skip_convs[i](s)
            skip = s + skip   #  skip_dim

            x = x + residual
            x = self.bn[i](x)

        x = F.relu(skip[:, :, :, -1:])
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x