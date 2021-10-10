import torch
from torch.utils import data
from Datasets.dataset_QNRF_JHU import Dataset
from Models.auxiliary_model import Model
import os
import argparse
checkpoint_logs_name = 'GCN_paper_JHU'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='JHU', type=str, help='dataset')
parser.add_argument('--data_path', default='path-to-your-data', type=str, help='path to dataset')
parser.add_argument('--load', default=True, action='store_true', help='load checkpoint')
parser.add_argument('--save_path', default='./checkpoints/' + checkpoint_logs_name, type=str, help='path to save checkpoint')
parser.add_argument('--gpu', default='0', type=str, help='gpu id')


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu



save_path = args.save_path + '/'
test_dataset = Dataset(args.data_path, args.dataset, False)
test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)

model = Model().cuda()


if args.load:
    checkpoint = torch.load(os.path.join(save_path, 'checkpoint_best.pth'))
    model.load_state_dict(checkpoint['model'])


model.eval()
with torch.no_grad():
    mae, mse = 0.0, 0.0

    for i, (image, gt, image_name) in enumerate(test_loader):
        B, C, W, H = image.size()
        scale_h = int(H / 128)
        scale_w = int(W / 128)
        patch_sum = 0

        for j in range(scale_h):
            for p in range(scale_w):
                patch = image[:, :, p * 128: (p + 1) * 128, j * 128: (j + 1) * 128].cuda()
                patch_pred, _, __ = model(patch)
                patch_sum += patch_pred.sum()
        mae += torch.abs(patch_sum - gt).item()
        mse += ((patch_sum - gt) ** 2).item()

    mae /= len(test_loader)
    mse /= len(test_loader)
    mse = mse ** 0.5
    print('MAE:', mae, 'MSE:', mse)


