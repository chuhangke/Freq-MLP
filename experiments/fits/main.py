import os
import argparse
import numpy as np

import sys

sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch

torch.set_num_threads(3)

from src.models.fits import FITS
from src.base.engine import BaseEngine
from src.utils.args import get_public_config
from src.utils.dataloader import load_dataset, load_adj_from_numpy, get_dataset_info
from src.utils.graph_algo import normalize_adj_mx
from src.utils.metrics import masked_mae
from src.utils.logging import get_logger



def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--lrate', type=float, default=1e-4)
    parser.add_argument('--wdecay', type=float, default=5e-4)
    parser.add_argument('--clip_grad_value', type=float, default=0)

    parser.add_argument('--input_len', type=int, default=48)
    parser.add_argument('--node_dim', type=int, default=32)
    parser.add_argument('--embed_dim', type=int, default=10)
    parser.add_argument('--output_len', type=int, default=48)
    parser.add_argument('--num_layer', type=int, default=2)
    parser.add_argument('--temp_dim_tid', type=int, default=32)
    parser.add_argument('--temp_dim_diw', type=int, default=32)
    parser.add_argument('--if_time_in_day', type=bool, default=True)
    parser.add_argument('--if_time_in_week', type=bool, default=True)
    parser.add_argument('--if_node', type=bool, default=True)
    parser.add_argument('--time_of_day_size', type=int, default=288)
    parser.add_argument('--day_of_week_size', type=int, default=7)
    parser.add_argument('--cut_freq', type=int, default=25)
    parser.add_argument('--individual', type=bool, default=False)
    #parser.add_argument('--device', type=str, default='cuda:0')
    #parser.add_argument('--dataset', type=str, default='SD')
    #parser.add_argument('--model_name', type=str, default='ours')
    args = parser.parse_args()

    log_dir = 'experiments/{}/{}/'.format(args.model_name, args.dataset)
    logger = get_logger(log_dir, __name__, 'record_s{}.log'.format(args.seed))
    logger.info(args)

    return args, log_dir, logger

def main():
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    device = torch.device(args.device)

    data_path, adj_path, node_num = get_dataset_info(args.dataset)
    logger.info('Adj path: ' + adj_path)

    dataloader, scaler = load_dataset(data_path, args, logger)

    model = FITS(cut_freq=args.cut_freq,
                 enc_in=node_num,
                 pred_len=args.output_len,
                 individual=args.individual,
                 seq_len=args.input_len
                 )

    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1,25], gamma=args.gamma)

    engine = BaseEngine(device=device,
                         model=model,
                         dataloader=dataloader,
                         scaler=scaler,
                         sampler=None,
                         loss_fn=loss_fn,
                         lrate=args.lrate,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         clip_grad_value=args.clip_grad_value,
                         max_epochs=args.max_epochs,
                         patience=args.patience,
                         log_dir=log_dir,
                         logger=logger,
                         seed=args.seed,
                         )

    if args.mode == 'train':
        engine.train()
    else:
        engine.evaluate(args.mode)

    #engine.evaluate('test')


if __name__ == "__main__":
    main()
