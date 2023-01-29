from .base_options import BaseOptions


class ExportOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--export_size', type=str, default='720,720', help='saves results here.')
        parser.add_argument('--phase', type=str, default='val', help='train, val, test, etc')

        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        self.isTrain = False
        return parser