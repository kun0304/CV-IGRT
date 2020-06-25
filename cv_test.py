# Coded by Xiaokun Liang
# E-mail: xiaokun@qq.com
# Stanford University


import numpy as np
import torch
import glob
import os
from param import *
from Functions import imshow3Dslice, generate_affine_matrix, SpatialTransform, extract_data_testing, save_img, save_checkerboard, load_cv_center, load_3d
from densenet3d import DenseNet3D


def test_main():
    # set the specific GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
    # load the pretrained model
    model = DenseNet3D(growthRate=4, depth=40, reduction=0.5, nClasses=6, bottleneck=True).cuda()
    model_name = sorted(glob.glob(model_dir + '/model*.pth'))
    model_name = model_name[len(model_name) - 1]
    model.load_state_dict(torch.load(model_name))

    transform = SpatialTransform().cuda()
    patient_dir = glob.glob(training_data_path + '/testing/*')

    for xi in range(0, len(patient_dir)):
        result_dir = patient_dir[xi] + '/result'
        if not os.path.isdir(result_dir):
            os.mkdir(result_dir)

        # load the preset location of the control volumes
        ct_name = glob.glob(patient_dir[xi] + '/ct_*.nii.gz')
        pct_name = ct_name[0]
        img_size = load_3d(pct_name).shape
        dct_names = ct_name[1:len(ct_name)]
        csv_names = glob.glob(patient_dir[xi] + '/*.csv')[0]
        cv_center_array = load_cv_center(csv_names)

        for yi in range(0, len(dct_names)):
            dct_name = dct_names[yi]

            # prediction for the different control volumes
            cv_with_align = torch.zeros(img_size)
            cv_without_align = torch.zeros(img_size)
            cv_pred = np.zeros((cv_center_array.shape[0], 6))
            for i_cv in range(0, cv_center_array.shape[0]):
                cv_location = cv_center_array[i_cv][:]
                dct, pct, pct_cv_annotate, pct_cv, pct_cv_binary = extract_data_testing(pct_name, dct_name, control_volume_size, cv_location)
                prediction_temp = model(dct, pct_cv_annotate)
                trans_dct_for_indiv_cv = torch.squeeze(transform(dct, prediction_temp)).cpu().detach()
                cv_with_align = cv_with_align + torch.mul(torch.squeeze(pct_cv_binary), trans_dct_for_indiv_cv)
                cv_without_align = cv_without_align + torch.mul(torch.squeeze(pct_cv_binary), torch.squeeze(dct).cpu().detach())
                prediction_temp = prediction_temp.cpu().detach().numpy()
                cv_pred[i_cv, :] = prediction_temp

            average_pred = torch.from_numpy(np.reshape(np.mean(cv_pred, 0), (1, 6))).cuda().float()
            # trans_dct = np.squeeze(transform(dct, average_pred, affine_matrix).cpu().detach().numpy())

            # save the result
            dct = np.squeeze(dct.cpu().detach().numpy())
            pct = np.squeeze(pct.cpu().detach().numpy())
            cv_with_align = np.squeeze(cv_with_align.cpu().detach().numpy())
            cv_without_align = np.squeeze(cv_without_align.cpu().detach().numpy())

            cv_with_binary = np.zeros(cv_with_align.shape)
            cv_with_binary[cv_with_align == 0.0] = 1.0
            cv_overlap_with_align = pct * cv_with_binary + (cv_with_align - 1.2 * np.std(pct)) * np.abs(1 - cv_with_binary)

            cv_without_binary = np.zeros(cv_without_align.shape)
            cv_without_binary[cv_without_align == 0.0] = 1.0
            cv_overlap_without_align = pct * cv_without_binary + (cv_without_align - 1.2 * np.std(pct)) * np.abs(
                1 - cv_without_binary)
            # imshow3Dslice(cv_overlap_pct)

            dct_case = dct_name[len(patient_dir[xi])+1:len(dct_name)-7]
            np.save(result_dir + '/' + dct_case + '_final_6dof.npy', average_pred.cpu().detach().numpy())
            np.save(result_dir + '/' + dct_case + '_cv_6dof.npy', cv_pred)
            # save_checkerboard(pct, dct, result_dir + '/' + dct_case + '_without_align.nii.gz')
            # save_checkerboard(pct, trans_dct, result_dir + '/' + dct_case + '_with_align.nii.gz')
            # save_img(trans_dct, result_dir + '/' + dct_case + '_trans_dct.nii.gz')
            save_img(cv_overlap_with_align, result_dir + '/' + dct_case + '_cv_overlap_pct_with_align.nii.gz')
            save_img(cv_overlap_without_align, result_dir + '/' + dct_case + '_cv_overlap_pct_without_align.nii.gz')


test_main()
