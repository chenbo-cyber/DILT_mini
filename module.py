import torch.nn as nn

def set_module(args):
    
    net = None
    if args.module_type == 'skip':
        net = SkipNet(signal_dim=args.signal_dim, n_filters=args.fr_n_filters,
                    inner_dim=args.fr_inner_dim, n_layers=args.fr_n_layers,
                    upsampling=int(args.label_size/args.fr_inner_dim), kernel_size=args.fr_kernel_size,
                    kernel_out=args.fr_kernel_out)

    net.cuda()

    return net


class SkipNet(nn.Module):
    def __init__(self, signal_dim=50, n_filters=8, n_layers=3, inner_dim=125,
                 kernel_size=3, upsampling=8, kernel_out=25):
        super().__init__()
        self.fr_size = inner_dim * upsampling
        self.n_filters = n_filters
        self.inner=inner_dim
        self.n_layers=n_layers
        self.in_layer1 = nn.Linear(signal_dim, inner_dim * (n_filters // 16), bias=False)
        self.in_layer2 = nn.Conv2d(1, n_filters // 8, kernel_size=(1, 3), padding=(0, 3//2), bias=False)
        mod=[]
        for i in range(self.n_layers):
            tmp = []
            tmp += [
                nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size //2, bias=False,
                          padding_mode='circular'),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
                nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size //2, bias=False,
                          padding_mode='circular'),
                nn.BatchNorm1d(n_filters),
            ]
            mod+= [nn.Sequential(*tmp)]
        self.mod=nn.Sequential(*mod)
        activate_layer = []
        for i in range(self.n_layers):
            activate_layer+=[nn.ReLU()]
        self.activate_layer=nn.Sequential(*activate_layer)

        # (280 - 1) * 2 - 2 * 17 + 35 + 1
        self.out_layer = nn.ConvTranspose1d(n_filters, 1, kernel_out, stride=upsampling,
                                            padding=(kernel_out - upsampling + 1) // 2, output_padding=1, bias=False)
    def forward(self, inp):
        bsz = inp.size(0)
        inp = inp.view(bsz, -1)
        # x1 = self.in_layer(inp).view(bsz, self.n_filters, -1)
        x1 = self.in_layer1(inp).view(bsz, 1, self.n_filters // 16, -1)
        x2 = self.in_layer2(x1).view(bsz, self.n_filters, -1)
        x3 = x2
        # x3 = self.in_layer3(x2).view(bsz, self.n_filters, -1)
    
        for i in range(self.n_layers):
            res_x = self.mod[i](x3)
            x3 = res_x + x3
            x = self.activate_layer[i](x3)
        x = self.out_layer(x).view(bsz, -1)
        return x