from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--lk_level', type=int, default=1, help='number of image levels for direct vo')
        self.parser.add_argument('--use_ssim', default=False, action='store_true', help='use ssim loss')
        self.parser.add_argument('--smooth_term', type=str, default='lap', help='smoothness term type, choose between lap, 1st, 2nd')
        self.parser.add_argument('--lambda_S', type=float, default=.01, help='smoothness cost weight')
        self.parser.add_argument('--lambda_E', type=float, default=.01, help='explainable mask regulariation cost weight')
        self.parser.add_argument('--epoch_num', type=int, default=20, help='number of epochs for training')
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=10, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=int, default=-1, help='which epoch to load? set to epoch number, set -1 to train from scratch')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.isTrain = True
