import cv2
import time
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from PIL.TiffTags import TAGS
import tifffile

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as f
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader


from rockedgesdetectors.pidinet.utils import cross_entropy_loss_RCF
from rockedgesdetectors import ModelPiDiNet
from rocknetmanager.dataset import Dataset

import matplotlib.pyplot as plt

from rocknetmanager.train import ModelTrain
from rocknetmanager.save_checkpoint import save_checkpoint


def main():
    path_lst = Path("D:/1.ToSaver/profileimages/train_data/train.lst")
    data = Dataset(path_lst=path_lst)

    seed = int(time.time())
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    checkpoint_path = Path("../models/table7_pidinet.pth")
    #checkpoint_path = Path("save_models/checkpoint_000.pth")
    test_image_folder = Path("../test_images")

    model = ModelPiDiNet(checkpoint_path)
    #model = ModelPiDiNet()

    conv_weights, bn_weights, relu_weights = model.get_weights()
    wd = 1e-3
    lr = 1e-4

    param_groups = [{
        'params': conv_weights,
        'weight_decay': wd,
        'lr': lr}, {
        'params': bn_weights,
        'weight_decay': 0.1 * wd,
        'lr': lr}, {
        'params': relu_weights,
        'weight_decay': 0.0,
        'lr': lr
    }]

    optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.99))
    for image_path in test_image_folder.iterdir():
        save_image_test(model, image_path)

    trainer = ModelTrain(data, model.model, optimizer)
    for epoch in range(1, 100):
        trainer.train()

        saveID = save_checkpoint({
            'epoch': epoch,
            'state_dict': model.model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, epoch, "../save_models")
        for image_path in test_image_folder.iterdir():
            epoch_folder = Path("../train_test/epoch_" + str(epoch))
            epoch_folder.mkdir(parents=False, exist_ok=True)
            save_image_test(model, image_path, epoch_folder)


def save_image_test(model, image_path: Path, save_folder=None):
    save_folder = "../train_test" if save_folder is None else str(save_folder)
    #
    model.model.eval()
    image = cv2.imread(str(image_path))
    result = model(image)
    fig = plt.figure(figsize=(7, 4))
    axs = [fig.add_subplot(1, 2, 1),
           fig.add_subplot(1, 2, 2)]
    axs[1].imshow(image)
    axs[0].imshow(result)
    #plt.show()
    fig.savefig(save_folder + "/" + image_path.name)


if __name__ == '__main__':
    main()
