class AFGCAttention(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(AFGCAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.fc = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.mix = Mix()

    def forward(self, input):
        x = self.avg_pool(input)
        x1 = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)#(1,64,1)
        x2 = self.fc(x).squeeze(-1).transpose(-1, -2)#(1,1,64)
        out1 = torch.sum(torch.matmul(x1,x2),dim=1).unsqueeze(-1).unsqueeze(-1)#(1,64,1,1)
        #x1 = x1.transpose(-1, -2).unsqueeze(-1)
        out1 = self.sigmoid(out1)
        out2 = torch.sum(torch.matmul(x2.transpose(-1, -2),x1.transpose(-1, -2)),dim=1).unsqueeze(-1).unsqueeze(-1)

        #out2 = self.fc(x)
        out2 = self.sigmoid(out2)
        out = self.mix(out1,out2)
        out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)

        return input*out



class EfficientBalanceFusionModuleV1(nn.Module):
    def __init__(self, inc) -> None:
        super().__init__()
        self.adjust_conv = nn.Identity()
        if inc[0] != inc[1]:
            self.adjust_conv = Conv(inc[0], inc[1], k=1)

        self.afgca = AFGCAttention(inc[1] * 2)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        x0, x1 = x
        x0 = self.adjust_conv(x0)

        x_concat = torch.cat([x0, x1], dim=1)
        x_concat = self.afgca(x_concat)

        x0_weight, x1_weight = torch.split(x_concat, [x0.size()[1], x1.size()[1]], dim=1)
        x0_weight = torch.sigmoid(x0_weight)
        x1_weight = torch.sigmoid(x1_weight)

        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)

        fused_x0 = alpha * x0 * (1 + x1_weight) + (1 - alpha) * x1 * x0_weight
        fused_x1 = beta * x1 * (1 + x0_weight) + (1 - beta) * x0 * x1_weight

        return torch.cat([fused_x0, fused_x1], dim=1)

