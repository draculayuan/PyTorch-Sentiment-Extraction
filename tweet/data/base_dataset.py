from torch.utils.data import Dataset

class BaseDataset(Dataset):

    """
    A dataset should implement
        1. __len__ to get size of the dataset, Required
        2. __getitem__ to get a single data, Required
        3. parse to parse list from file, Required
    """

    def __init__(self):
        super(BaseDataset, self).__init__()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def parse(self, img_root, list_path):
        raise NotImplementedError
