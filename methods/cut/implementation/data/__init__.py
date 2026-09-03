"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
from data.base_dataset import BaseDataset


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt, rank=None, world_size=None):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Parameters:
        opt -- training options
        rank (int, optional) -- process rank for DDP (if None, uses regular DataLoader)
        world_size (int, optional) -- total number of processes for DDP

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
        >>> dataset = create_dataset(opt, rank=0, world_size=2)  # for DDP
    """
    data_loader = CustomDatasetDataLoader(opt, rank=rank, world_size=world_size)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt, rank=None, world_size=None):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.

        Parameters:
            opt -- training options
            rank (int, optional) -- process rank for DDP (if None, uses regular DataLoader)
            world_size (int, optional) -- total number of processes for DDP
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        if rank is None:
            print("dataset [%s] was created" % type(self.dataset).__name__)
        else:
            if rank == 0:
                print("dataset [%s] was created (DDP mode, rank %d/%d)" % (type(self.dataset).__name__, rank, world_size))
        
        # Use DistributedSampler if rank and world_size are provided
        if rank is not None and world_size is not None:
            sampler = DistributedSampler(
                self.dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=not opt.serial_batches
            )
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batch_size,
                sampler=sampler,
                num_workers=int(opt.num_threads),
                drop_last=True if opt.isTrain else False,
                pin_memory=True,
            )
            self.sampler = sampler
        else:
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batch_size,
                shuffle=not opt.serial_batches,
                num_workers=int(opt.num_threads),
                drop_last=True if opt.isTrain else False,
            )
            self.sampler = None

    def set_epoch(self, epoch):
        self.dataset.current_epoch = epoch
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
