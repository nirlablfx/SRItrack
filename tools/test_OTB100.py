# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np
import math
import sys
import shutil
sys.path.append('')

from pysot.core.config import cfg
from pysot.tracker.sri_tracker import SRITracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from pysot.models.model_builder import ModelBuilder

from toolkit.datasets import DatasetFactory

parser = argparse.ArgumentParser(description='SRItrack tracking')

parser.add_argument('--video', default='', type=str,
                    
        help='eval one special video')
parser.add_argument('--dataset', type=str, default='OTB100',
        help='datasets')#OTB100 LaSOT UAV123 GOT10K
parser.add_argument('--vis', action='store_true',default=True,
        help='whether visualzie result')
parser.add_argument('--snapshot', type=str, default='./snapshot/checkpoint_e20.pth',
        help='snapshot of models to eval')

parser.add_argument('--config', type=str, default='',
        help='config file')

args = parser.parse_args()

torch.set_num_threads(1)


def main():
    # load config
    cfg.merge_from_file(args.config)

    # hp_search
    params = getattr(cfg.HP_SEARCH,args.dataset)
    hp = {'lr': params[0], 'penalty_k':params[1], 'window_lr':params[2]}


    dataset_root = ''

# 
    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = SRITracker(model, cfg.TRACK)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = 'snapshot'+args.snapshot[-6:-4]

    # OPE tracking
    for v_idx, video in enumerate(dataset):
        if args.video != '':
            # test one special video
            if video.name != args.video:
                continue
        toc = 0
        pred_bboxes = []
        # track_times = []
        for idx, (img, gt_bbox) in enumerate(video):  #img  360,480,3   1080,1920,3  说明图片大小不需要一致
            tic = cv2.getTickCount()
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                tracker.init(img, gt_bbox_)
                pred_bbox = gt_bbox_
                pred_bboxes.append(pred_bbox)
            else:
                outputs = tracker.track(img, hp)  #outputs:bbox
                pred_bbox = outputs['bbox']
                pred_bboxes.append(pred_bbox)
            toc += cv2.getTickCount() - tic

            if idx == 0:
                cv2.destroyAllWindows()
            if args.vis and idx > 0:
                if not any(map(math.isnan,gt_bbox)):
                    gt_bbox = list(map(int, gt_bbox))
                    pred_bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                  (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
        toc /= cv2.getTickFrequency()
        # save results
        model_path = os.path.join('results', args.dataset, model_name)
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        result_path = os.path.join(model_path, '{}.txt'.format(video.name))
        with open(result_path, 'w') as f:
            for x in pred_bboxes:
                f.write(','.join([str(i) for i in x])+'\n')
        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            v_idx+1, video.name, toc, idx / toc))
    os.chdir(model_path)
    save_file = '../%s' % dataset
    shutil.make_archive(save_file, 'zip')
    print('Records saved at', save_file + '.zip')


if __name__ == '__main__':
    main()
