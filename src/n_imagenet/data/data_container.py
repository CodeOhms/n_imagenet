from n_imagenet.data.imagenet import ImageNetDataset
from torch.utils.data import DataLoader, Sampler
from n_imagenet.base.data.data_container import DataContainer
import torch


class ImageNetContainer(DataContainer):
    def __init__(self, cfg, ds_transform=None, **kwargs):
        """
        cfg contains all the contents of a config file
        """
        super(ImageNetContainer, self).__init__(cfg)
        print(f'Initializing data container {self.__class__.__name__}...')
        self.ds_transform = ds_transform
        self.gen_dataset()
        self.gen_dataloader()

    def gen_dataset(self, **kwargs):
        dataset_name = getattr(self.cfg, 'dataset_name', 'imagenet')
        if dataset_name == 'imagenet':
            if self.cfg.mode == 'train':

                train_dataset = ImageNetDataset(self.cfg, mode='train', transform=self.ds_transform)
                val_dataset = ImageNetDataset(self.cfg, mode='val', transform=self.ds_transform)

                self.dataset['train'] = train_dataset
                self.dataset['val'] = val_dataset

            elif self.cfg.mode == 'test':
                if getattr(self.cfg, 'streak', False):
                    test_dataset = StreakImageNetDataset(self.cfg)
                    self.dataset['test'] = test_dataset
                else:
                    test_dataset = ImageNetDataset(self.cfg, mode='test', transform=self.ds_transform)
                    self.dataset['test'] = test_dataset

            else:
                raise AttributeError('Mode not provided')

    def _handle_dataloader_kwargs(self, **kwargs):
        dataloader_def_kwargs = {
            "collate_fn": self.dict_collate_fn,
            "batch_size": self.cfg.batch_size,
            "num_workers": self.cfg.num_workers,
            "drop_last": False,
            "pin_memory": self.cfg.pin_memory
        }
        for kword, default_karg in dataloader_def_kwargs.items():
            if kword not in kwargs.keys():
                kwargs[kword] = default_karg
        return kwargs

    def gen_train_dataloader(self, **kwargs):
        assert self.dataloader is not None
        kwargs = self._handle_dataloader_kwargs(**kwargs)
        if kwargs['sampler'] is not None:
            assert isinstance(kwargs['sampler'], Sampler)
        else:
            kwargs['shuffle'] = True
        self.dataloader['train'] = DataLoader(self.dataset['train'], **kwargs)

    def gen_val_dataloader(self, **kwargs):
        assert self.dataloader is not None
        kwargs = self._handle_dataloader_kwargs(**kwargs)
        if kwargs['sampler'] is not None:
            assert isinstance(kwargs['sampler'], Sampler)
        else:
            kwargs['shuffle'] = True
        self.dataloader['val'] = DataLoader(self.dataset['val'], **kwargs)

    def gen_test_dataloader(self, **kwargs):
        assert self.dataloader is not None
        kwargs = self._handle_dataloader_kwargs(**kwargs)
        kwargs['shuffle'] = False
        self.dataloader['test'] = DataLoader(self.dataset['test'], **kwargs)

    def gen_dataloader(self, **kwargs):
        self.gen_all_dataloaders(**kwargs)

    def gen_all_dataloaders(self, train_kwargs={}, val_kwargs={}, test_kwargs={}, **kwargs):
        assert self.dataloader is not None
        if self.cfg.mode == 'train':
            self.gen_train_dataloader(**train_kwargs)
            self.gen_val_dataloader(**val_kwargs)
        elif self.cfg.mode == 'test':
            self.gen_test_dataloader(**test_kwargs)
        else:
            raise AttributeError('Mode not provided')

    # collate_fn for generating dictionary-style batches.
    def dict_collate_fn(self, list_data):
        # list_data is as follows: [(img: torch.Tensor(H, W, 3), event: torch.Tensor(H, W, 4), label: int), ...]
        collate_type = getattr(self.cfg, 'collate_type', 'normal')
        if collate_type == 'normal':
            events, labels = list(zip(*list_data))

            event_batch = torch.stack(events, dim=0)
            label_batch = torch.LongTensor(labels)
            return {
                'input_data': event_batch,
                'label': label_batch
            }
