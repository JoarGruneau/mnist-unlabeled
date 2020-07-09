from torchvision import datasets, transforms
from torch.utils import data

base_transform_list = [
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081, ))
]

augumentation_tranform_list = [
    transforms.RandomAffine(degrees=45,
                            translate=(0.1, 0.1),
                            scale=(0.8, 1.2),
                            shear=[(-20, 20), (-10, 10)])
]

AUGUMENTATION_TRANSFORM = transforms.Compose(base_transform_list +
                                             augumentation_tranform_list)

CLEAN_TRANSFORM = transforms.Compose(base_transform_list)


def get_data_generators(batch_size):
    train = datasets.MNIST(root='./data',
                           train=True,
                           download=True,
                           transform=CLEAN_TRANSFORM)

    test = datasets.MNIST(root='./data',
                          train=False,
                          download=True,
                          transform=CLEAN_TRANSFORM)

    train_generator = data.DataLoader(train,
                                      shuffle=True,
                                      num_workers=8,
                                      batch_size=batch_size)

    test_generator = data.DataLoader(train,
                                     shuffle=False,
                                     num_workers=8,
                                     batch_size=batch_size)

    return train_generator, test_generator
