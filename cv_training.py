# Coded by Xiaokun Liang
# E-mail: xiaokun@qq.com
# Stanford University

import torch
import os
import sys
import numpy as np
import timeit
from param import *
from Functions import imshow3Dslice,load_pretrain_model, SpatialTransform, extract_data_training, extract_moved_cv, data_load_cv_project, generate_affine_matrix, ncc_loss,update_lr
from densenet3d import DenseNet3D


def train_main():
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # set the specific GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index

    # Formulate the network
    model = DenseNet3D(growthRate=4, depth=25, reduction=0.5, nClasses=6, bottleneck=True)
    if torch.cuda.is_available():
        model = model.cuda()

    # Load the ct dataset
    ct_dataset, image_size = data_load_cv_project(training_data_path)
    val_ct_dataset, image_size = data_load_cv_project(validation_data_path)

    transform = SpatialTransform()
    if torch.cuda.is_available():
        transform = transform.cuda()

    # if there is a pretrained model, load it.
    current_iter, curr_lr, optimizer, model, lossall, val_lossall = load_pretrain_model(model_dir, lr, model, num_iter, learning_rate_decay_times, decay_rate)

    # main training step
    for step in range(current_iter, num_iter):
        start1 = timeit.default_timer()
        # data extract in the specific training step
        dct, pct, pct_cv_annotate, pct_cv, cv_position = extract_data_training(ct_dataset, batch_size, control_volume_size)

        # input the data into the network

        prediction = model(dct, pct_cv_annotate)

        # transformed the dCT
        trans_dct = transform(dct, prediction)

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

        # validation
        with torch.no_grad():
            val_dct, val_pct, val_pct_cv_annotate, val_pct_cv, val_cv_position = extract_data_training(val_ct_dataset, 1,
                                                                                   control_volume_size)
            val_prediction = model(val_dct, val_pct_cv_annotate)
            val_trans_dct = transform(val_dct, val_prediction)
            val_trans_dct_cv = extract_moved_cv(val_trans_dct, val_cv_position, control_volume_size)
            val_global_loss = ncc_loss(val_trans_dct, val_pct)
            val_cv_loss = ncc_loss(val_trans_dct_cv, val_pct_cv)
            val_loss = val_global_loss + lambda_cv * val_cv_loss

        # show and save the training loss
        lossall[:, step] = np.array([loss.item(), global_loss.item(), cv_loss.item()])
        val_lossall[:, step] = np.array([val_loss.item(), val_global_loss.item(), val_cv_loss.item()])

        # update the learning rate
        if (step + 1) % (int(num_iter/learning_rate_decay_times)) == 0:
            curr_lr = decay_rate * curr_lr
            update_lr(optimizer, curr_lr)

        start2 = timeit.default_timer()

        # save the loss value
        np.save(model_dir + '//training_loss_totalIter_' + str(num_iter) + '.npy', lossall)
        np.save(model_dir + '//val_loss_totalIter_' + str(num_iter) + '.npy', val_lossall)

        # save the checkpoint model during training & displace the loss
        if (((step + 1) % save_net_nIter) == 0) or (step + 1 == num_iter):
            print('**************************************************')
            model_name1 = model_dir + '/model_iter_' + str(step + 1).rjust(6, '0') + '.pth'
            torch.save(model.state_dict(), model_name1)
            sys.stdout.write(
                "\r" + ' step "{0}" -> compound "{1:.4f}" - global_ncc "{2:.4f}" - cv ncc "{3:.4f}" - '.format(
                    step+1, np.mean(lossall[0, step - save_net_nIter + 1:step + 1]), np.mean(lossall[1, step - save_net_nIter + 1:step + 1]), np.mean(lossall[2, step - save_net_nIter + 1:step + 1])))
            sys.stdout.flush()
            print('training: It still takes: ' + str((num_iter - step) * (start2 - start1) / 60) + ' mins')


train_main()
