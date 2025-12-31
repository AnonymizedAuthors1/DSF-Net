from skimage import measure
import numpy as np
import torch
import torch.nn.functional as Func
from torch import nn


def deep_uncertain_loss(output_fg, output_bg, output_uc, gaze, t1, t2):
    gaze_binary = torch.where(gaze < t1, 0, gaze)
    gaze_binary = torch.where(gaze_binary > t2, 1, gaze_binary)
    mask = torch.where((gaze_binary >= t1) & (gaze_binary <= t2), 0, 1)
    gaze_binary_m, output_m = gaze_binary * mask, output_fg * mask
    certain_num = torch.sum(mask, dim=(1, 2, 3))
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    loss_fg = criterion(output_m, gaze_binary_m.float())
    loss_fg = (torch.sum(loss_fg, dim=(1, 2, 3)) + 1e-5) / (certain_num + 1e-5)
    loss_fg = loss_fg.mean()

    gaze_binary = torch.where(gaze < t1, -1, gaze)
    gaze_binary = torch.where(gaze_binary > t2, 0, gaze_binary)
    gaze_binary = torch.where(gaze_binary == -1, 1, gaze_binary)
    gaze_binary_m, output_m = gaze_binary * mask, output_bg * mask
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    loss_bg = criterion(output_m, gaze_binary_m.float())
    loss_bg = (torch.sum(loss_bg, dim=(1, 2, 3)) + 1e-5) / (certain_num + 1e-5)
    loss_bg = loss_bg.mean()

    output_fg, output_bg, output_uc = nn.Sigmoid()(output_fg), nn.Sigmoid()(output_bg), nn.Sigmoid()(output_uc)
    loss_uc = (((output_fg * output_bg).sum() + (output_fg * output_uc).sum() + (output_bg * output_uc).sum()) /
               (output_fg.size(0) * output_fg.size(2) * output_fg.size(3)))

    return loss_fg.mean() + loss_bg.mean() + 0.5 * loss_uc


def cross_entropy_loss_uncertain(output1, gaze, t1, t2):
    gaze_binary = torch.where(gaze < t1, 0, gaze)
    gaze_binary = torch.where(gaze_binary > t2, 1, gaze_binary)
    mask = torch.where((gaze_binary >= t1) & (gaze_binary <= t2), 0, 1)
    gaze_binary_m, output1_m = gaze_binary * mask, output1 * mask
    certain_num = torch.sum(mask, dim=(1, 2, 3))

    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    ce_loss_1 = criterion(output1_m, gaze_binary_m.float())
    ce_loss_1 = (torch.sum(ce_loss_1, dim=(1, 2, 3)) + 1e-5) / (certain_num + 1e-5)
    ce_loss = ce_loss_1.mean()
    return ce_loss

def intergrity_loss(pred):
    predictions_original_list = []
    pred = nn.Sigmoid()(pred)
    pred = torch.concat([1-pred, pred], dim=1)
    for i in range(pred.shape[0]):
        prediction = np.uint8(np.argmax(pred[i, :, :, :].detach().cpu(), axis=0))
        prediction = keep_largest_connected_components(prediction)
        prediction = torch.from_numpy(prediction).to(pred.device)
        predictions_original_list.append(prediction)
    predictions = torch.stack(predictions_original_list)
    pred_keep_largest_connected = torch.unsqueeze(predictions, 1)

    loss_integrity = 1 - Func.cosine_similarity(pred[:, 1, :, :], pred_keep_largest_connected, dim=1).mean()
    return loss_integrity, pred_keep_largest_connected


def keep_largest_connected_components(mask):
    '''
    Keeps only the largest connected components of each label for a segmentation mask.
    '''
    # keep a heart connectivity
    mask_shape = mask.shape

    heart_slice = np.where((mask > 0), 1, 0)
    out_heart = np.zeros(heart_slice.shape, dtype=np.uint8)
    for struc_id in [1]:
        binary_img = heart_slice == struc_id
        blobs = measure.label(binary_img, connectivity=1)
        props = measure.regionprops(blobs)
        if not props:
            continue
        area = [ele.area for ele in props]
        largest_blob_ind = np.argmax(area)
        largest_blob_label = props[largest_blob_ind].label
        out_heart[blobs == largest_blob_label] = struc_id

    out_img = np.zeros(mask.shape, dtype=np.uint8)
    for struc_id in [1, 2, 3]:
        binary_img = mask == struc_id
        blobs = measure.label(binary_img, connectivity=1)
        props = measure.regionprops(blobs)
        if not props:
            continue
        area = [ele.area for ele in props]
        largest_blob_ind = np.argmax(area)
        largest_blob_label = props[largest_blob_ind].label
        out_img[blobs == largest_blob_label] = struc_id

    final_img = out_heart * out_img
    return final_img


