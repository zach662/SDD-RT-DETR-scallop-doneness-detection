class WiseDIoULoss(torch.nn.Module):
    momentum = 1e-2
    alpha = 1.7
    delta = 2.7

    def __init__(self, ltype='DIoU', monotonous=False, inner_iou=False, focaler_iou=False):
        super().__init__()
        assert getattr(self, f'_{ltype}', None), f'The loss function {ltype} does not exist'
        self.ltype = ltype
        self.monotonous = monotonous
        self.inner_iou = inner_iou
        self.focaler_iou = focaler_iou
        self.register_buffer('iou_mean', torch.tensor(1.))

    def __getitem__(self, item):
        if callable(self._fget[item]):
            self._fget[item] = self._fget[item]()
        return self._fget[item]

    def forward(self, pred, target, ret_iou=False, ratio=1.0, d=0.0, u=0.95, **kwargs):
        self._fget = {
            # pred, target: x0,y0,x1,y1
            'pred': self._xywh2xyxy(pred),
            'target': self._xywh2xyxy(target),
            # x,y,w,h
            'pred_xy': pred[..., :2],
            'pred_wh': pred[..., 2:],
            'target_xy': target[..., :2],
            'target_wh': target[..., 2:],
            # x0,y0,x1,y1
            'min_coord': lambda: torch.minimum(self['pred'][..., :4], self['target'][..., :4]),
            'max_coord': lambda: torch.maximum(self['pred'][..., :4], self['target'][..., :4]),
            # The overlapping region
            'wh_inter': lambda: torch.relu(self['min_coord'][..., 2: 4] - self['max_coord'][..., :2]),
            's_inter': lambda: torch.prod(self['wh_inter'], dim=-1),
            # The area covered
            's_union': lambda: torch.prod(self['pred_wh'], dim=-1) +
                               torch.prod(self['target_wh'], dim=-1) - self['s_inter'],
            # The smallest enclosing box
            'wh_box': lambda: self['max_coord'][..., 2: 4] - self['min_coord'][..., :2],
            's_box': lambda: torch.prod(self['wh_box'], dim=-1),
            'l2_box': lambda: torch.square(self['wh_box']).sum(dim=-1),
            # The central points' connection of the bounding boxes
            'd_center': lambda: self['pred_xy'] - self['target_xy'],
            'l2_center': lambda: torch.square(self['d_center']).sum(dim=-1),
            # IoU / Inner-IoU / Focaler-IoU
            'iou': lambda: (1 - get_inner_iou(pred, target, xywh=False, ratio=ratio).squeeze()) if self.inner_iou else (1 - ((self['s_inter'] / self['s_union'] - d) / (u - d)).clamp(0, 1) if self.focaler_iou else 1 - self['s_inter'] / self['s_union']),
        }

        if self.training:
            self.iou_mean = self.iou_mean.to(self['iou'].device)
            self.iou_mean.mul_(1 - self.momentum)
            self.iou_mean.add_(self.momentum * self['iou'].detach().mean())

        ret = self._scaled_loss(getattr(self, f'_{self.ltype}')(**kwargs)), self['iou']
        delattr(self, '_fget')
        return ret if ret_iou else ret[0]

    def _DIoU(self):
        return self['iou'] + self['l2_center'] / self['l2_box']
