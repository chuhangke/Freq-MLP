import os
import argparse
import numpy as np

import sys

sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch

torch.set_num_threads(3)

from src.models.STAEformer import STAEformer
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
    parser.add_argument('--Kt', type=int, default=3)
    parser.add_argument('--Ks', type=int, default=3)
    parser.add_argument('--block_num', type=int, default=2)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.5)

    parser.add_argument('--lrate', type=float, default=2e-3)
    parser.add_argument('--wdecay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--clip_grad_value', type=float, default=0)

    parser.add_argument('--input_len', type=int, default=12)
    parser.add_argument('--node_dim', type=int, default=32)
    # parser.add_argument('--input_dim', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=24)
    parser.add_argument('--output_len', type=int, default=12)
    parser.add_argument('--num_layer', type=int, default=3)
    parser.add_argument('--temp_dim_tid', type=int, default=24)
    parser.add_argument('--temp_dim_diw', type=int, default=24)
    parser.add_argument('--if_time_in_day', type=bool, default=True)
    parser.add_argument('--if_time_in_week', type=bool, default=True)
    parser.add_argument('--use_mixed_proj', type=bool, default=True)
    parser.add_argument('--time_of_day_size', type=int, default=288)
    parser.add_argument('--day_of_week_size', type=int, default=7)
    parser.add_argument('--beta', type=float, default=0.001)
    # parser.add_argument('--device', type=str, default='cuda:0')
    # parser.add_argument('--dataset', type=str, default='SD')
    # parser.add_argument('--model_name', type=str, default='ours')
    args = parser.parse_args()

    log_dir = './experiments/{}/{}/'.format(args.model_name, args.dataset)
    logger = get_logger(log_dir, __name__, 'record_s{}.log'.format(args.seed))
    logger.info(args)

    return args, log_dir, logger


def main():
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    device = torch.device(0)

    data_path, adj_path, node_num = get_dataset_info(args.dataset)
    logger.info('Adj path: ' + adj_path)

    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = adj_mx - np.eye(node_num)

    gso = normalize_adj_mx(adj_mx, 'scalap')[0]
    gso = torch.tensor(gso).to(device)

    Ko = args.seq_len - (args.Kt - 1) * 2 * args.block_num

    dataloader, scaler = load_dataset(data_path, args, logger)

    model = STAEformer(num_nodes=node_num,
                       input_len=args.input_len,
                       output_len=args.output_len,
                       in_steps=args.input_len,
                       out_steps=args.output_len,
                       steps_per_day=args.time_of_day_size,
                       input_dim=3,
                       output_dim=1,
                       input_embedding_dim=args.embed_dim,
                       tod_embedding_dim=args.temp_dim_tid,
                       dow_embedding_dim=args.temp_dim_diw,
                       spatial_embedding_dim=0,
                       adaptive_embedding_dim=80,
                       feed_forward_dim=256,
                       num_heads=4,
                       num_layers=args.num_layer,
                       dropout=args.dropout,
                       use_mixed_proj=args.use_mixed_proj,
                       )

    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

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


if __name__ == "__main__":
    main()
