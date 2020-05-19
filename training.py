# -*- coding: UTF-8 -*-
# Coded by Xiaokun Liang
# E-mail: xiaokun@qq.com

import torch
import os
import sys
import numpy as np
import timeit
from param import *
from Functions import load_pretrain_model, SpatialTransform, extract_data_training, extract_moved_cv, data_load_cv_project, generate_grid, ncc_loss,update_lr
from densenet3d import DenseNet3D


def train_main():
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # set the specific GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index

    # Formulate the network
    model = DenseNet3D(growthRate=4, depth=8, reduction=0.5, nClasses=6, bottleneck=True).cuda()

    # Load the dataset
    pct_dataset, dct_dataset, image_size = data_load_cv_project(training_data_path)

    # generate the sample grid for transform the image
    sample_grid = generate_grid(image_size)

    transform = SpatialTransform().cuda()

    # create the matrix for saving the training loss
    lossall = np.zeros((3, num_iter))

    # if there is a pretrained model, load it.
    current_iter, curr_lr, optimizer, model = load_pretrain_model(model_dir, lr, model, num_iter, learning_rate_decay_times, decay_rate)

    # main training step
    for step in range(current_iter, num_iter):
        start1 = timeit.default_timer()
        # data extract in the specific training step
        dct, pct, pct_cv_annotate, pct_cv, cv_position = extract_data_training(pct_dataset, dct_dataset, batch_size, control_volume_size)
        # input the data into the network
        prediction = model(dct, pct_cv_annotate)

        # transformed the dCT
        trans_dct = transform(dct, prediction, sample_grid)

        # extract the predicted cv from the trans_dct
        trans_dct_cv = extract_moved_cv(trans_dct, cv_position, control_volume_size)

        # calculate the individual loss
        global_loss = ncc_loss(trans_dct, pct)
        cv_loss = ncc_loss(trans_dct_cv, pct_cv)

        # calculate the compound loss
        loss = global_loss + lambda_cv * cv_loss

        # back propagation of the model
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # back propagation, compute gradients
        optimizer.step()  # apply gradients

        # show and save the training loss
        lossall[:, step] = np.array([loss.item(), global_loss.item(), cv_loss.item()])

        # update the learning rate
        if (step + 1) % (int(num_iter/learning_rate_decay_times)) == 0:
            curr_lr = decay_rate * curr_lr
            update_lr(optimizer, curr_lr)

        start2 = timeit.default_timer()
        # save the checkpoint model during training & displace the loss
        if (((step + 1) % save_net_nIter) == 0) or (step + 1 == num_iter):
            print('**************************************************')
            np.save(model_dir + '//training_loss_totalIter_' + str(num_iter) + '.npy', lossall)
            model_name1 = model_dir + '/model_iter_' + str(step + 1).rjust(6, '0') + '.pth'
            torch.save(model.state_dict(), model_name1)
            sys.stdout.write(
                "\r" + ' step "{0}" -> compound "{1:.4f}" - global_ncc "{2:.4f}" - cv ncc "{3:.4f}" - '.format(
                    step+1, np.mean(lossall[0, step - save_net_nIter + 1:step + 1]), np.mean(lossall[1, step - save_net_nIter + 1:step + 1]), np.mean(lossall[2, step - save_net_nIter + 1:step + 1])))
            sys.stdout.flush()

            print('training: It still takes: ' + str((num_iter - step) * (start2 - start1) / 60) + ' mins')


train_main()
