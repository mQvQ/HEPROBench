from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=None, help='window id of the web display. Default is random window id')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_per_iteration', type=int, default=10000, help='frequency of saving checkpoints per iteration')
        parser.add_argument('--evaluation_freq', type=int, default=10000, help='evaluation freq')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--continue_iteration', type=int, default=0, help='continue training from this iteration')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--pretrained_name', type=str, default=None, help='resume training from another checkpoint')

        # training parameters
        parser.add_argument('--total_iterations', type=int, default=100000, help='total training iterations (default: 100000)')
        parser.add_argument('--n_epochs', type=int, default=200, help='[DEPRECATED] number of epochs with the initial learning rate (kept for compatibility)')
        parser.add_argument('--n_epochs_decay', type=int, default=200, help='[DEPRECATED] number of epochs to linearly decay learning rate to zero (kept for compatibility)')
        parser.add_argument('--n_iters', type=int, default=50000, help='number of iterations with the initial learning rate (for linear LR policy)')
        parser.add_argument('--n_iters_decay', type=int, default=50000, help='number of iterations to linearly decay learning rate to zero (for linear LR policy)')
        parser.add_argument('--lr_update_freq', type=int, default=1, help='frequency of updating learning rate (every N iterations)')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

        # DDP parameters
        parser.add_argument('--ddp_enabled', action='store_true', help='enable DDP training')
        parser.add_argument('--ddp_backend', type=str, default='nccl', help='DDP backend (nccl, gloo)')
        parser.add_argument('--ddp_init_method', type=str, default='env://', help='DDP initialization method')
        parser.add_argument('--ddp_master_addr', type=str, default='localhost', help='DDP master address')
        parser.add_argument('--ddp_master_port', type=str, default='29508', help='DDP master port')
        parser.add_argument('--ddp_timeout_sec', type=int, default=3600, help='process group timeout in seconds (increase if checkpointing or dataloading is slow)')

        self.isTrain = True
        return parser
