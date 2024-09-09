# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.models import backbone
from pysot.models.loss_car import make_siamcar_loss_evaluator
from pysot.models.backbone import get_backbone
from pysot.models.head.car_head import CARHead
from pysot.models.neck import get_neck

from pysot.utils.location_grid import compute_locations
from pysot.models.neck import get_neck,build_fpn

from pysot.models.BSDM import BSDM
from pysot.models.SPRI import SPRI

from pysot.utils.xcorr import xcorr_depthwise

class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone             #resnet50
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,   
                                     **cfg.BACKBONE.KWARGS)
        # self.backbone =Models.alexnet(pretrained = True)
        
        # build adjust layer
        if cfg.ADJUST.ADJUST:# True   
            self.neck = get_neck(cfg.ADJUST.TYPE,            #AdjustAllLayer
                                 **cfg.ADJUST.KWARGS)        #各层的输入通道数和输出通道数

        # build car head
        self.car_head = CARHead(cfg, 256)

        self.xcorr_depthwise = xcorr_depthwise #定义深度互相关

        self.loss_evaluator = make_siamcar_loss_evaluator(cfg)  #设计loss函数

        self.down = nn.ConvTranspose2d(256 * 3, 256, 1, 1) #转置卷积 类似于上采样

        self.bsdm = BSDM(n_channels=256, n_classes=256)
        self.bsdm1 = BSDM(n_channels=384, n_classes=384)
        self.fpn = build_fpn(cfg)

        self.spri = SPRI(num_blocks=2,num_features=256)


    def template(self, z):
        zf = self.backbone(z)
        # zf = self.split(zf) #10 256 7 7
        zf[0] = self.bsdm1(zf[0])
        zf[1] = self.bsdm1(zf[1])
        zf[2] = self.bsdm(zf[2])
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)

        zf = self.fpn(zf)  # 3*8*8*7*7#12,256,7,7
        
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        # xf = self.split (xf) # 10 256 31 31
        xf[0] = self.bsdm1(xf[0])
        xf[1] = self.bsdm1(xf[1])
        xf[2] = self.bsdm(xf[2])
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)

        xf = self.fpn(xf)  # 8*8*31*31#12,256,31,31


        features = self.xcorr_depthwise(xf[0],self.zf[0])
        for i in range(len(xf)-1):
            features_new = self.xcorr_depthwise(xf[i+1],self.zf[i+1])
            features = torch.cat([features,features_new],1)
        features = self.down(features)
        features = self.spri(features)
        features = F.avg_pool2d(features, kernel_size=2, stride=2)


        cls, loc, cen = self.car_head(features)
        return {
                'cls': cls,
                'loc': loc,
                'cen': cen,
                'fea': loc
               }

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()#20 25 25
        label_loc = data['bbox'].cuda()#20 4

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        zf[0] = self.bsdm1(zf[0])
        zf[1] = self.bsdm1(zf[1])
        zf[2] = self.bsdm(zf[2])
        xf[0] = self.bsdm1(xf[0])
        xf[1] = self.bsdm1(xf[1])
        xf[2] = self.bsdm(xf[2])

        if cfg.ADJUST.ADJUST:#True
            zf = self.neck(zf)
            xf = self.neck(xf)
            xf=self.fpn(xf) #8*8*31*31#12,256,31,31
            zf=self.fpn(zf) #3*8*8*7*7#12,256,7,7


        features = self.xcorr_depthwise(xf[0],zf[0])
        for i in range(len(xf)-1):
            features_new = self.xcorr_depthwise(xf[i+1],zf[i+1])
            features = torch.cat([features,features_new],1)
        features = self.down(features)
        features = self.spri(features)
        features = F.avg_pool2d(features, kernel_size=2, stride=2)

        cls, loc, cen = self.car_head(features)


        locations = compute_locations(cls, cfg.TRACK.STRIDE)#625 2
        cls = self.log_softmax(
            cls)
        cls_loss, loc_loss, cen_loss = self.loss_evaluator(
            locations,
            cls,#20, 1, 25, 25, 2
            loc,#20 4 25 25
            cen, label_cls, label_loc
        )#20 1 25 25   20 25 25   20 4

        # get loss
        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss + cfg.TRAIN.CEN_WEIGHT * cen_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['cen_loss'] = cen_loss
        return outputs
