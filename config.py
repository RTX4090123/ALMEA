import os.path as osp
import argparse
import platform

class config():
    def __init__(self):
        self.this_dir = osp.dirname(__file__)
        self.data_root = osp.abspath(osp.join(self.this_dir, '..', 'data', ''))

    def get_args(self):
        parser = argparse.ArgumentParser()
        """
        primary parameters setting
        """
        parser.add_argument('--gpu', default=0, type=int)
        parser.add_argument('--eval_epoch', default=2, type=int, help='evaluate each n epoch')
        parser.add_argument("--only_test", default=0, type=int, choices=[0, 1])
        parser.add_argument("--data_split", default='norm', type=str, help="Experiment split")
        parser.add_argument("--data_rate", type=float, default=0.2, help="training set rate")
        parser.add_argument('--epoch', default=100, type=int)
        parser.add_argument('--epoch_per_CYCLES', default=50, type=int)
        parser.add_argument('--lr', type=float, default= 1e-3)
        parser.add_argument('--mask', type=float, default=0.1, help='mask probability')
        parser.add_argument("--scheduler", default="cos", type=str, choices=["linear", "cos", "fixed"])
        parser.add_argument("--optim", default="adam", type=str, choices=["adamw", "adam"])
        parser.add_argument("--rho", default=0.1, type=float, help='Lagrange penalty term')
        parser.add_argument("--alpha_", default=0.15, type=float, help='Sparsification strength')
        parser.add_argument("--strategy", default="frobenius", type=str, help='Sparsification method')
        parser.add_argument("--tau3", default=0.15, type=float, help='truncation value for most valuable potential pair')
        parser.add_argument("--early_stop_threshold", default=1e-7, type=float, help='alternate directions for early stopping')
        parser.add_argument('--CYCLES', default=2, type=int, help='Number of active learning strategy activations. i.e. If CYCLES = 1, it means that the active learning strategy is activated 1 time')
        parser.add_argument('--batch_size', default=3500, type=int)
        parser.add_argument("--csls", action="store_true", default=False, help="use CSLS for inference")
        parser.add_argument("--csls_k", type=int, default=3, help="top k for csls")
        parser.add_argument("--data_choice", default="FBDB15K", type=str, choices=["FBYG15K", "FBDB15K"], help="Experiment path")
        parser.add_argument("--random_seed", default=42, type=int)
        parser.add_argument("--exp_id", default="seed_42", type=str, help="Experiment ID")
        parser.add_argument('--workers', type=int, default=12)
        parser.add_argument('--dist', type=int, default=0, help='whether to dist')
        parser.add_argument('--accumulation_steps', type=int, default=1)
        parser.add_argument("--ratio", type=str, default="1.0", help="which visual adapt",
                            choices=["0.05", "0.1", "0.15", "0.2", "0.3", "0.4",
                                     "0.45", "0.5", "0.55", "0.6", "0.7", "0.75", "0.8", "0.9", "1.0"])
        parser.add_argument("--num_layers", type=int, default=3, help='VAE layers')

        """
        secondary parameters setting
        """
        parser.add_argument("--unsup_mode", type=str, default="img", help="unsup mode", choices=["img", "name", "char"])
        parser.add_argument("--unsup_k", type=int, default=1000, help="|visual seed|")
        parser.add_argument("--il_start", type=int, default=500, help="If Il, when to start?")
        parser.add_argument("--no_tensorboard", default=False, action="store_true")
        parser.add_argument("--exp_name", default="EA_exp", type=str, help="Experiment name")
        parser.add_argument("--dump_path", default="dump/", type=str, help="Experiment dump path")
        parser.add_argument("--data_path", default="mmkg", type=str, help="Experiment path")
        parser.add_argument("--unsup", action="store_true", default=False)
        parser.add_argument("--word_embedding", type=str, default="glove", help="the type of word embedding, [glove|fasttext]", choices=["glove", "bert"])
        parser.add_argument("--es", action="store_true", default=False, help="process the datasets for entity synthesis")
        parser.add_argument('--clip', type=float, default=1., help='gradient clipping')

        parser.add_argument("--w_name", action="store_false", default=True, help="with name features")
        parser.add_argument("--w_char", action="store_false", default=True, help="with char features")


        parser.add_argument('--rank', type=int, default=0, help='rank to dist')
        parser.add_argument("--distance", type=int, default=2, help="L1 distance or L2 distance. ('1', '2')", choices=[1, 2])


        parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
        parser.add_argument('--world-size', default=3, type=int,
                            help='number of distributed processes')
        parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
        parser.add_argument("--il", action="store_true", default=False, help="Iterative learning?")
        parser.add_argument('--weight_decay', type=float, default=0.0001)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        self.cfg = parser.parse_args()

    def update_configs(self):
        self.cfg.data_root = self.data_root
        self.cfg.data_path = osp.join(self.data_root, self.cfg.data_path)

        if self.cfg.data_choice in ["FBYG15K", "FBDB15K"]:
            self.cfg.data_split = "norm"
            self.cfg.w_name = False
            self.cfg.w_char = False
            data_split_name = f"{self.cfg.data_rate}_"

        self.cfg.exp_id = f"{'my-solution'}_{self.cfg.data_choice}_{data_split_name}{self.cfg.exp_id}"
        self.cfg.dump_path = osp.join(self.cfg.data_path, self.cfg.dump_path)
        return self.cfg
