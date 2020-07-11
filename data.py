from torchvision import datasets, transforms
import torch

augumentation_transform = transforms.Compose([
    transforms.RandomRotation((-10, 10),
                              resample=False,
                              expand=False,
                              center=None),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=(0, 0),
                            translate=(0, 0.1),
                            scale=None,
                            shear=None,
                            resample=False,
                            fillcolor=0),
    transforms.RandomAffine(degrees=(0, 0),
                            translate=None,
                            scale=(0.9, 1.1),
                            shear=None,
                            resample=False,
                            fillcolor=0),
    transforms.RandomAffine(degrees=(0, 0),
                            translate=None,
                            scale=None,
                            shear=(-10, 5),
                            resample=False,
                            fillcolor=0),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, )),
])

clean_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, ), (0.5, ))])


def get_data_generators(batch_size):
    train = datasets.MNIST(root='./data',
                           train=True,
                           download=True,
                           transform=clean_transform)

    train_aug = datasets.MNIST(root='./data',
                               train=True,
                               download=True,
                               transform=augumentation_transform)

    train_set = Dataset(train, train_aug)
    labeled_data = get_labeled_data(train)

    test = datasets.MNIST(root='./data',
                          train=False,
                          download=True,
                          transform=clean_transform)

    test_aug = datasets.MNIST(root='./data',
                              train=False,
                              download=True,
                              transform=augumentation_transform)

    test_set = Dataset(test, test_aug)

    train_generator = torch.utils.data.DataLoader(train_set,
                                                  shuffle=True,
                                                  num_workers=8,
                                                  batch_size=batch_size)

    test_generator = torch.utils.data.DataLoader(test_set,
                                                 shuffle=False,
                                                 num_workers=8,
                                                 batch_size=int(len(test)/10))

    return labeled_data, train_generator, test_generator


def get_labeled_data(data):
    labels = []
    labeled_data = []
    for i in range(len(data)):
        if data[i][1] not in labels:
            labeled_data.append(data[i][0])
            labels.append(data[i][1])
        if len(labels) == 10:
            return torch.stack(labeled_data, dim=0), torch.tensor(labels)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, aug_data):
        self.data = data
        self.aug_data = aug_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        x_aug, _ = self.aug_data[idx]
        return (x, x_aug), y
