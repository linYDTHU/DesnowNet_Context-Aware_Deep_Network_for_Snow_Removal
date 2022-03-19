import os
import random
import torch
from torchvision import transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
from PIL import Image


class snow_dataset(data.Dataset):

    def __init__(self, gt_root, mask_root, synthetic_root, is_crop=True):
        self.gt_root = gt_root
        self.mask_root = mask_root
        self.synthetic_root = synthetic_root
        self.is_crop = is_crop

        self.imgs_list = os.listdir(gt_root)

    def __getitem__(self, index):
        img_name = self.imgs_list[index]
        gt_path = os.path.join(self.gt_root, img_name)
        mask_path = os.path.join(self.mask_root, img_name)
        synthetic_path = os.path.join(self.synthetic_root, img_name)

        # read images
        gt_data = Image.open(gt_path).convert('RGB')
        mask_data = Image.open(mask_path).convert('L')
        synthetic_data = Image.open(synthetic_path).convert('RGB')

        # totensor and random crop
        toTensor = transforms.ToTensor()
        gt_tensor = toTensor(gt_data)
        mask_tensor = toTensor(mask_data)
        synthetic_tensor = toTensor(synthetic_data)

        if self.is_crop:
            h, w = gt_tensor.shape[1:]
            y = random.randint(0, h - 64)
            x = random.randint(0, w - 64)
            gt_tensor = gt_tensor[:, y:y + 64, x:x + 64]
            mask_tensor = mask_tensor[:, y:y + 64, x:x + 64]
            synthetic_tensor = synthetic_tensor[:, y:y + 64, x:x + 64]

        return gt_tensor, mask_tensor, synthetic_tensor

    def __len__(self):
        return len(self.imgs_list)

class inference_dataset(data.Dataset):

    def __init__(self, root):
        self.root = root
        self.imgs_list = os.listdir(root)

    def __getitem__(self, index):
        img_name = self.imgs_list[index]
        img_path = os.path.join(self.root, img_name)

        # read images
        img_data = Image.open(img_path).convert('RGB')

        # totensor
        toTensor = transforms.ToTensor()
        img_tensor = toTensor(img_data)

        return img_tensor, img_name

    def __len__(self):
        return len(self.imgs_list)

if __name__ == '__main__':
    gt_root = "../Snow100K-S/gt"
    mask_root = "../Snow100K-S/mask"
    synthetic_root = "../Snow100K-S/synthetic"
    dataset = snow_dataset(gt_root, mask_root, synthetic_root, is_crop=True)
    img1, img2, img3 = dataset.__getitem__(1)
    ToPILImage = transforms.ToPILImage()
    img1_ = ToPILImage(img1)
    img2_ = ToPILImage(img2)
    img3_ = ToPILImage(img3)
    figure, axes = plt.subplots(1,3)
    plt.subplot(1,3,1)
    plt.imshow(img1_)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,3,2)
    plt.imshow(img2_,cmap=plt.cm.gray)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,3,3)
    plt.imshow(img3_)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    print("finished")
    # root = "../Snow100K-S/synthetic"
    # dataset = inference_dataset(root)
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
    #                                           shuffle=True,
    #                                           num_workers=6,
    #                                           pin_memory=True)
    # for i, data in enumerate(data_loader):
    #     img, name = data
    #     if i==5:
    #         break
    # ToPILImage = transforms.ToPILImage()
    # img_ = ToPILImage(img)
    # plt.imshow(img_)
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()
    # print("finished")
