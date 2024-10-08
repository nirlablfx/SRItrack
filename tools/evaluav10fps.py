import os
import sys
import time
import argparse
import functools
sys.path.append('')

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool

from toolkit.datasets import UAV10Dataset
from toolkit.evaluation import OPEBenchmark
from toolkit.visualization import draw_success_precision

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single Object Tracking Evaluation')
    parser.add_argument('--dataset_dir', default='',type=str, help='dataset root directory')
    parser.add_argument('--dataset', default='UAV123_10fps',type=str, help='dataset name')
    parser.add_argument('--tracker_result_dir',default='./results', type=str, help='tracker result root')
    parser.add_argument('--trackers',default='general_model', nargs='+')
    parser.add_argument('--vis', default='false',dest='vis', action='store_true')
    parser.add_argument('--show_video_level', default='True',dest='show_video_level', action='store_true')
    parser.add_argument('--tracker_prefix', '-t', default='snapshot',type=str, help='tracker name')
    parser.add_argument('--num', default=1, type=int, help='number of processes to eval')
    # parser.set_defaults(show_video_level=False)
    args = parser.parse_args()

    tracker_dir = os.path.join(args.tracker_result_dir, args.dataset)
    trackers = glob(os.path.join(args.tracker_result_dir,
                                  args.dataset,
                                  args.tracker_prefix+'*'))
                                #   '*'))
    trackers = [x.split('/')[-1] for x in trackers]


    root = os.path.realpath(os.path.join(os.path.dirname(__file__),
                             '../testing_dataset'))
    root = os.path.join(root, args.dataset)


    trackers = [os.path.basename(x) for x in trackers]

  
    assert len(trackers) > 0
    args.num = min(args.num, len(trackers))

    if 'UAV123_10fps' in args.dataset:
        dataset = UAV10Dataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=18):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=18):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                show_video_level=args.show_video_level)
