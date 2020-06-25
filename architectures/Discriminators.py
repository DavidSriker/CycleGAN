from architectures.ArchitecturesUtils import *

class Dis(nn.Module):
    def __init__(self, in_channels, out_channels, n_filters=64):
        super(Dis, self).__init__()

        self.d1 = DiscriminatorLayer(in_channels, n_filters, 4, normalization=False)
        self.d2 = DiscriminatorLayer(n_filters, n_filters * 2, 4)
        self.d3 = DiscriminatorLayer(n_filters * 2, n_filters * 4, 4)
        self.d4 = DiscriminatorLayer(n_filters * 4, n_filters * 8, 4)
        self.reflection_pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(n_filters * 8, out_channels, kernel_size=3, stride=1)

    def forward(self, x):
        out = self.d1(x)
        out = self.d2(out)
        out = self.d3(out)
        out = self.d4(out)
        out = self.conv(out)
        return self.reflection_pad(out)


if __name__ == '__main__':
    print("Test Dis")
    input_c = 1
    output_c = 1

    device = torch.device(("cpu", "cuda")[torch.cuda.is_available()])
    D = Dis(in_channels=input_c, out_channels=output_c)
    x = torch.zeros((6, input_c, 256, 256), dtype=torch.float32)
    print("input shape = ", x.shape)
    y = D(x.to(device))
    print("output shape = ", y.shape)