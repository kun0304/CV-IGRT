import SimpleITK as sItk
import numpy as np
import torch
import glob
import timeit
import csv
import torch.nn as nn


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def load_pretrain_model(model_dir, lr, model, num_iter, learning_rate_decay_times, decay_rate):
    curr_lr = lr
    model_name = sorted(glob.glob(model_dir + '/model*.pth'))
    try:
        model_name = model_name[len(model_name) - 1]
        model.load_state_dict(torch.load(model_name))
        current_iter = int(model_name[len(model_name) - 10:len(model_name) - 4])
        optimizer = torch.optim.Adam(model.parameters(), lr=curr_lr)
        for iter in range(0, current_iter):
            if (iter + 1) % (int(num_iter / learning_rate_decay_times)) == 0:
                curr_lr = decay_rate * curr_lr
                update_lr(optimizer, curr_lr)
    except:
        print('no pretrained model!')
        current_iter = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=curr_lr)
    return current_iter, curr_lr, optimizer, model


def load_cv_center(csv_names):
    with open(csv_names[0], newline='') as camile:
        data = list(csv.reader(camile))
    cv_center_array = np.zeros([data.__len__(), 3], dtype=int)
    for ixxx in range(0, data.__len__()):
        cv_center_array[ixxx][0:3] = data[ixxx][1:4]
    del data
    return cv_center_array


def extract_moved_cv(moved, cv_cord_ini, control_volume_size):
    moved_cv = torch.zeros((moved.shape[0],) + (1,) + control_volume_size).cuda().float()
    for batch_index in range(0, moved.shape[0]):
        x_ini = int(cv_cord_ini[batch_index][0])
        y_ini = int(cv_cord_ini[batch_index][1])
        z_ini = int(cv_cord_ini[batch_index][2])
        moved_cv[batch_index, :, :, :, :] = moved[batch_index, :, x_ini:x_ini + control_volume_size[0],
                                                              y_ini:y_ini + control_volume_size[1],
                                                              z_ini:z_ini + control_volume_size[2]]
    return moved_cv


def data_load_cv_project(data_path):
    pct_sum = []
    dct_sum = []
    patient_id_name = glob.glob(data_path + '/training/*')
    for patient_ID_index in range(0, len(patient_id_name)):
        start1 = timeit.default_timer()
        dct_sum.append([])
        pct_sum.append([])
        name_pCT = glob.glob(patient_id_name[patient_ID_index] + '/pCT*.gz')
        pct_sum[patient_ID_index].append(Norm_Zscore(imgnorm(load_3d(name_pCT[0]))))
        name_fraction = glob.glob(patient_id_name[patient_ID_index] + '/dCT*.gz')
        for fraction_index in range(0, len(name_fraction)):
            dct_sum[patient_ID_index].append(Norm_Zscore(imgnorm(load_3d(name_fraction[fraction_index]))))
        start2 = timeit.default_timer()
        print('DataLoad: it still takes: ' + str((len(patient_id_name) - patient_ID_index)
                                                 * (start2 - start1) / 60) + ' minutes')
    img_size = pct_sum[0][0].shape
    return pct_sum, dct_sum, img_size


def exclude_impossi_cv(fixed1, control_volume_size):
    fixed_mask = 1 - (fixed1 < (np.std(fixed1)))
    percent_object = 0
    # Generate the batch position and crop the ROI for training
    while percent_object < 10:
        x_ini = int(np.random.randint(0, fixed1.shape[0] - control_volume_size[0], 1))
        y_ini = int(np.random.randint(0, fixed1.shape[1] - control_volume_size[1], 1))
        z_ini = int(np.random.randint(0, fixed1.shape[2] - control_volume_size[2], 1))

        num_ele = control_volume_size[0] * control_volume_size[1] * control_volume_size[2]
        roi_mask = fixed_mask[x_ini:x_ini + control_volume_size[0],
                   y_ini:y_ini + control_volume_size[1],
                   z_ini:z_ini + control_volume_size[2]]
        percent_object = np.sum(roi_mask) / num_ele * 100
    return x_ini, y_ini, z_ini


def extract_data_training(pct_dataset, dct_dataset, batch_size, control_volume_size):
    image_size = pct_dataset[0][0].shape
    dct = np.zeros((batch_size,) + image_size)
    pct = np.zeros((batch_size,) + image_size)
    pct_cv_annotate = np.zeros((batch_size,) + image_size)
    pct_cv = np.zeros((batch_size,) + control_volume_size)
    cv_cord_ini = np.zeros((batch_size, 3))
    for batch_index in range(0, batch_size):
        patient_ID_index = np.random.permutation(len(pct_dataset))[0]
        fraction_ID_index = np.random.permutation(len(dct_dataset[patient_ID_index]))[0]
        dct[batch_index, :, :, :] = dct_dataset[patient_ID_index][fraction_ID_index]
        pct[batch_index, :, :, :] = pct_dataset[patient_ID_index][0]
        pct_cv_annotate[batch_index, :, :, :] = pct_dataset[patient_ID_index][0]

        x_ini, y_ini, z_ini = exclude_impossi_cv(pct[batch_index, :, :, :], control_volume_size)

        pct_cv[batch_index, :, :, :] = pct[batch_index, x_ini:x_ini + control_volume_size[0], y_ini:y_ini + control_volume_size[1],
        z_ini:z_ini + control_volume_size[2]]

        pct_cv_annotate[batch_index, x_ini:x_ini + control_volume_size[0], y_ini:y_ini + control_volume_size[1],
        z_ini:z_ini + control_volume_size[2]] = 10.0 * pct_cv_annotate[batch_index, x_ini:x_ini + control_volume_size[0], y_ini:y_ini + control_volume_size[1],
                   z_ini:z_ini + control_volume_size[2]]
        cv_cord_ini[batch_index, :] = [x_ini, y_ini, z_ini]

    dct = np.reshape(dct, (dct.shape[0],) + (1,) + dct.shape[1:4])
    pct = np.reshape(pct, (pct.shape[0],) + (1,) + pct.shape[1:4])
    pct_cv = np.reshape(pct_cv, (pct_cv.shape[0],) + (1,) + pct_cv.shape[1:4])
    pct_cv_annotate = np.reshape(pct_cv_annotate, (pct_cv_annotate.shape[0],) + (1,) + pct_cv_annotate.shape[1:4])

    dct = torch.from_numpy(dct).cuda().float()
    pct = torch.from_numpy(pct).cuda().float()
    pct_cv = torch.from_numpy(pct_cv).cuda().float()
    pct_cv_annotate = torch.from_numpy(pct_cv_annotate).cuda().float()
    return dct, pct, pct_cv_annotate, pct_cv, cv_cord_ini


def extract_data_testing(pct_name, dct_name, control_volume_size, cv_center_array):
    pct = Norm_Zscore(imgnorm(load_3d(pct_name)))
    dct = Norm_Zscore(imgnorm(load_3d(dct_name)))

    x_ini = int(cv_center_array[2] - control_volume_size[2]/2)
    y_ini = int(cv_center_array[1] - control_volume_size[1]/2)
    z_ini = int(cv_center_array[0] - control_volume_size[0]/2)

    pct_cv = pct[x_ini:x_ini + control_volume_size[0], y_ini:y_ini + control_volume_size[1],
    z_ini:z_ini + control_volume_size[2]]

    pct_cv_annotate = Norm_Zscore(imgnorm(load_3d(pct_name)))
    pct_cv_annotate[x_ini:x_ini + control_volume_size[0], y_ini:y_ini + control_volume_size[1],
    z_ini:z_ini + control_volume_size[2]] = 10.0 * pct_cv_annotate[x_ini:x_ini + control_volume_size[0], y_ini:y_ini + control_volume_size[1],
               z_ini:z_ini + control_volume_size[2]]

    dct = torch.from_numpy(np.reshape(dct, (1,) + (1,) + dct.shape)).cuda().float()
    pct = torch.from_numpy(np.reshape(pct, (1,) + (1,) + pct.shape)).cuda().float()
    pct_cv = torch.from_numpy(np.reshape(pct_cv, (1,) + (1,) + pct_cv.shape)).cuda().float()
    pct_cv_annotate = torch.from_numpy(np.reshape(pct_cv_annotate, (1,) + (1,) + pct_cv_annotate.shape)).cuda().float()
    return dct, pct, pct_cv_annotate, pct_cv


def ncc_loss(y_true_f, y_pred_f):
    mu = torch.mean(y_true_f)
    sig = torch.std(y_true_f)
    y_true_f = torch.div((y_true_f - mu), sig)
    y_true_f = y_true_f.contiguous().view(-1).cuda()

    mu = torch.mean(y_pred_f)
    sig = torch.std(y_pred_f)
    y_pred_f = torch.div((y_pred_f - mu), sig)
    y_pred_f = y_pred_f.contiguous().view(-1).cuda()

    xxx = torch.sum(torch.mul(y_true_f, y_pred_f))
    nn = (torch.sum(torch.mul(y_true_f, y_true_f)))
    mm = (torch.sum(torch.mul(y_pred_f, y_pred_f)))
    yyy = torch.sqrt(torch.mul(nn, mm))
    ncc = - torch.div(xxx, yyy)
    return ncc


def generate_grid(imgshape):
    x = np.arange(imgshape[0])
    y = np.arange(imgshape[1])
    z = np.arange(imgshape[2])
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid, 0, 2)
    grid = np.swapaxes(grid, 1, 2)

    grid = torch.from_numpy(grid).cuda().float()
    grid[:, :, :, 0] = (grid[:, :, :, 0] - ((imgshape[2] - 1) / 2)) / (imgshape[2] - 1) * 2
    grid[:, :, :, 1] = (grid[:, :, :, 1] - ((imgshape[1] - 1) / 2)) / (imgshape[1] - 1) * 2
    grid[:, :, :, 2] = (grid[:, :, :, 2] - ((imgshape[0] - 1) / 2)) / (imgshape[0] - 1) * 2
    return grid


def load_3d(name):
    X = sItk.GetArrayFromImage(sItk.ReadImage(name, sItk.sitkFloat32))
    return X


def imgnorm(N_I, index1=0.0001, index2=0.0001):
    I_sort = np.sort(N_I.flatten())
    I_min = I_sort[int(index1*len(I_sort))]
    I_max = I_sort[-int(index2*len(I_sort))]
    
    N_I = 1.0*(N_I-I_min)/(I_max-I_min)
    N_I[N_I > 1.0] = 1.0
    N_I[N_I < 0.0] = 0.0
    N_I2 = N_I.astype(np.float32)
    return N_I2


def Norm_Zscore(img):
    img = (img-np.mean(img))/np.std(img)
    return img


def save_img(I_img,savename):
    I2 = sItk.GetImageFromArray(I_img, isVector=False)
    sItk.WriteImage(I2, savename)


def generate_cb_templat(image_shape):
    img_temp = np.zeros(image_shape)
    img_temp[0:round(image_shape[0] / 2),
            0:round(image_shape[1] / 2),
            0:round(image_shape[2] / 2)] = 1
    img_temp[round(image_shape[0] / 2):image_shape[0],
            round(image_shape[1] / 2):image_shape[1],
            0:round(image_shape[2] / 2)] = 1
    img_temp[0:round(image_shape[0] / 2),
            round(image_shape[1] / 2):image_shape[1],
            round(image_shape[2] / 2):image_shape[2]] = 1

    img_temp[round(image_shape[0] / 2):image_shape[0],
            0:round(image_shape[1] / 2),
            round(image_shape[2] / 2):image_shape[2]] = 1

    return img_temp


def save_checkerboard(I_img1, I_img2, savename):
    cb_temp = generate_cb_templat(I_img1.shape)
    I_img = cb_temp * I_img1 + (1-cb_temp) * (I_img2 + np.std(I_img2))
    I2 = sItk.GetImageFromArray(I_img, isVector=False)
    sItk.WriteImage(I2, savename)


class SpatialTransform(nn.Module):
    def __init__(self):
        super(SpatialTransform, self).__init__()
    def forward(self, moving, prediction,sample_grid):
        grid_temp = torch.zeros((moving.shape[0],) + sample_grid.shape).cuda().float()
        for batch_index in range(0, moving.shape[0]):
            grid_temp[batch_index, :, :, :, :] = sample_grid
            c1 = torch.cos(prediction[batch_index][3] * (np.pi / 180.0))
            c2 = torch.cos(prediction[batch_index][4] * (np.pi / 180.0))
            c3 = torch.cos(prediction[batch_index][5] * (np.pi / 180.0))
            s1 = torch.sin(prediction[batch_index][3] * (np.pi / 180.0))
            s2 = torch.sin(prediction[batch_index][4] * (np.pi / 180.0))
            s3 = torch.sin(prediction[batch_index][5] * (np.pi / 180.0))

            R_zyx = torch.tensor([[c1 * c2, c1 * s2 * s3 - c3 * s1, s1 * s3 + c1 * c3 * s2],
                                  [c2 * s1, c1 * c3 + s1 * s2 * s3, c3 * s1 * s2 - c1 * s3],
                                  [-s2, c2 * s3, c2 * c3]])

            temp = torch.empty(len(grid_temp[batch_index, :, :, :, 0].contiguous().view(-1)), 3)
            temp[:, 0] = grid_temp[batch_index, :, :, :, 0].contiguous().view(-1)
            temp[:, 1] = grid_temp[batch_index, :, :, :, 1].contiguous().view(-1)
            temp[:, 2] = grid_temp[batch_index, :, :, :, 2].contiguous().view(-1)
            R_zyx = torch.transpose(R_zyx, 0, 1)

            temp = torch.mm(temp, R_zyx)

            grid_temp[batch_index, :, :, :, 0] = temp[:, 0].view(grid_temp.shape[1:4])
            grid_temp[batch_index, :, :, :, 1] = temp[:, 1].view(grid_temp.shape[1:4])
            grid_temp[batch_index, :, :, :, 2] = temp[:, 2].view(grid_temp.shape[1:4])

            grid_temp[batch_index, :, :, :, 0] = grid_temp[batch_index, :, :, :, 0] + 2 * (
                        prediction[batch_index][0] / grid_temp.size()[3])
            grid_temp[batch_index, :, :, :, 1] = grid_temp[batch_index, :, :, :, 1] + 2 * (
                        prediction[batch_index][1] / grid_temp.size()[3])
            grid_temp[batch_index, :, :, :, 2] = grid_temp[batch_index, :, :, :, 2] + 2 * (
                        prediction[batch_index][2] / grid_temp.size()[3])

        prediction = torch.nn.functional.grid_sample(moving, grid_temp, mode='bilinear', padding_mode="border")
        return prediction