# -*- coding: UTF-8 -*-
# Coded by Xiaokun Liang
# E-mail: xiaokun@qq.com

from param import *
import glob
import os
import numpy as np
import torch
from Functions import generate_grid, SpatialTransform, extract_data_testing, save_img, save_checkerboard, load_cv_center, load_3d
from densenet3d import DenseNet3D


def test_main():
    # load the pretrained model
    model = DenseNet3D(growthRate=4, depth=8, reduction=0.5, nClasses=6, bottleneck=True).cuda()
    model_name = sorted(glob.glob(model_dir + '/model*.pth'))
    model_name = model_name[len(model_name) - 1]
    model.load_state_dict(torch.load(model_name))

    transform = SpatialTransform().cuda()

    patient_dir = glob.glob(training_data_path + '/testing/*')

    image_size = load_3d(glob.glob(patient_dir[0] + '/pCT*.gz')[0]).shape
    sample_grid = generate_grid(image_size)

    for xi in range(0, len(patient_dir)):
        result_dir = patient_dir[xi] + '/result'
        if not os.path.isdir(result_dir):
            os.mkdir(result_dir)
        pct_name = glob.glob(patient_dir[xi] + '/pCT*.gz')[0]
        dct_names = glob.glob(patient_dir[xi] + '/dCT*.gz')
        csv_names = glob.glob(patient_dir[xi] + '/*.csv')
        cv_center_array = load_cv_center(csv_names)

        for yi in range(0, len(dct_names)):
            dct_name = dct_names[yi]
            cv_pred = np.zeros((cv_center_array.shape[0], 6))
            for i_cv in range(0, cv_center_array.shape[0]):
                cv_location = cv_center_array[i_cv][:]
                dct, pct, pct_cv_annotate, pct_cv = extract_data_testing(pct_name, dct_name, control_volume_size, cv_location)
                prediction_temp = model(dct, pct_cv_annotate)
                prediction_temp = prediction_temp.cpu().detach().numpy()
                cv_pred[i_cv, :] = prediction_temp
            average_pred = torch.from_numpy(np.reshape(np.mean(cv_pred, 0), (1, 6))).cuda().float()

            trans_dct = np.squeeze(transform(dct, average_pred, sample_grid).cpu().detach().numpy())

            dct = np.squeeze(dct.cpu().detach().numpy())
            pct = np.squeeze(pct.cpu().detach().numpy())

            dct_case = dct_name[len(patient_dir[xi])+1:len(dct_name)-7]
            np.save(result_dir + '/' + dct_case + '_final_6dof.npy', average_pred.cpu().detach().numpy())
            np.save(result_dir + '/' + dct_case + '_cv_6dof.npy', cv_pred)
            save_checkerboard(pct, dct, result_dir + '/' + dct_case + '_without_align.nii.gz')
            save_checkerboard(pct, trans_dct, result_dir + '/' + dct_case + '_with_align.nii.gz')
            save_img(trans_dct, result_dir + '/' + dct_case + '_trans_dct.nii.gz')


test_main()
