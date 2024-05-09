import torch
import numpy as np
import matplotlib.pyplot as plt


def figure_results(val_loader, train_loader, model, device):
    gt, pan, ms, p_tilde, u_tilde = next(iter(val_loader))
    gt_t, pan_t, ms_t, p_tilde_t, u_tilde_t = next(iter(train_loader))

    with torch.no_grad():
        xout = model.forward(ms.to(device).float(), pan.to(device).float(), p_tilde.to(device), u_tilde.to(device)).cpu().detach()
        xout_t = model.forward(ms_t.to(device).float(), pan_t.to(device).float(), p_tilde_t.to(device), u_tilde_t.to(device)).cpu().detach()

    figure_val = plot_model(gt, ms, pan, xout)
    figure_train = plot_model(gt_t, ms_t, pan_t, xout_t)

    return {"Validation": figure_val, "Training": figure_train}


def plot_model(gt, ms, pan, out):
    N = gt.shape[0] if gt.shape[0] <= 5 else 5

    width = 5
    height = 5

    fig, axs = plt.subplots(N, 5, figsize=(5 * width, N * height))

    if N != 1:
        axs[0][0].set_title("PAN image")
        axs[0][1].set_title("MS image")
        axs[0][2].set_title("Output image")
        axs[0][3].set_title("Ground truth")
        axs[0][4].set_title("Error image")

        for axs_row, x_gt, x_ms, x_pan, x_out in zip(axs, gt, ms, pan, out):
            for ax in axs_row:
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

            valor_min = np.min(np.array([np.min(np.array(x_pan.cpu().detach())[0:3, :, :]),
                                         np.min(np.array(x_ms.cpu().detach())[0:3, :, :]),
                                         np.min(np.array(x_out.cpu().detach())[0:3, :, :]),
                                         np.min(np.array(x_gt.cpu().detach())[0:3, :, :])]))
            valor_d = np.max(np.array([np.max(np.array(x_pan.cpu().detach())[0:3, :, :]),
                                       np.max(np.array(x_ms.cpu().detach())[0:3, :, :]),
                                       np.max(np.array(x_out.cpu().detach())[0:3, :, :]),
                                       np.max(np.array(x_gt.cpu().detach())[0:3, :, :])])) - valor_min

            pan = Tensor2B4B3B2(x_pan, valor_d, valor_min)
            axs_row[0].imshow(pan, vmin=0, vmax=1)

            ms = Tensor2B4B3B2(x_ms, valor_d, valor_min)
            axs_row[1].imshow(ms, vmin=0, vmax=1)

            out = Tensor2B4B3B2(x_out, valor_d, valor_min)
            axs_row[2].imshow(out, vmin=0, vmax=1)

            gt = Tensor2B4B3B2(x_gt, valor_d, valor_min)
            axs_row[3].imshow(gt, vmin=0, vmax=1)

            error = RGB_error(x_out, x_gt)
            axs_row[4].imshow(error, vmin=0, vmax=1)

            [ax.axis("off") for ax in axs_row]
    else:
        axs[0].set_title("PAN image")
        axs[1].set_title("MS image")
        axs[2].set_title("Output image")
        axs[3].set_title("Ground truth")
        axs[4].set_title("Error image")
        for x_gt, x_ms, x_pan, x_out in zip(gt, ms, pan, out):
            valor_min = np.min(np.array([np.min(np.array(x_pan.cpu().detach())[0:3, :, :]),
                                         np.min(np.array(x_ms.cpu().detach())[0:3, :, :]),
                                         np.min(np.array(x_out.cpu().detach())[0:3, :, :]),
                                         np.min(np.array(x_gt.cpu().detach())[0:3, :, :])]))
            valor_d = np.max(np.array([np.max(np.array(x_pan.cpu().detach())[0:3, :, :]),
                                       np.max(np.array(x_ms.cpu().detach())[0:3, :, :]),
                                       np.max(np.array(x_out.cpu().detach())[0:3, :, :]),
                                       np.max(np.array(x_gt.cpu().detach())[0:3, :, :])])) - valor_min

            pan = Tensor2B4B3B2(x_pan, valor_d, valor_min)
            axs[0].imshow(pan, vmin=0, vmax=1)

            ms = Tensor2B4B3B2(x_ms, valor_d, valor_min)
            axs[1].imshow(ms, vmin=0, vmax=1)

            out = Tensor2B4B3B2(x_out, valor_d, valor_min)
            axs[2].imshow(out, vmin=0, vmax=1)

            gt = Tensor2B4B3B2(x_gt, valor_d, valor_min)
            axs[3].imshow(gt, vmin=0, vmax=1)

            error = RGB_error(x_out, x_gt)
            axs[4].imshow(error, vmin=0, vmax=1)

            [ax.axis("off") for ax in axs]

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_8_channels(gt, ms, pan, out):
    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    size = 256 * px
    fig, axs = plt.subplots(4, 5, figsize=(5 * size, 4 * size))
    axs[0][0].set_title('PAN')
    axs[0][1].set_title('SEN2')
    axs[0][2].set_title('PRED')
    axs[0][3].set_title('VENUS')
    axs[0][4].set_title('ERROR')

    axs[0][0].set_ylabel('B4B3B2', rotation=0, size='large', labelpad=25)
    axs[1][0].set_ylabel('B7B6B5', rotation=0, size='large', labelpad=25)
    axs[2][0].set_ylabel('B8', rotation=0, size='large', labelpad=15)
    axs[3][0].set_ylabel('B8A', rotation=0, size='large', labelpad=15)

    for x_gt, x_ms, x_pan, x_out in zip(gt, ms, pan, out):
        valor_min = np.min(np.array([np.min(np.array(x_pan.cpu().detach())[0:3, :, :]),
                                     np.min(np.array(x_ms.cpu().detach())[0:3, :, :]),
                                     np.min(np.array(x_out.cpu().detach())[0:3, :, :]),
                                     np.min(np.array(x_gt.cpu().detach())[0:3, :, :])]))
        valor_d = np.max(np.array([np.max(np.array(x_pan.cpu().detach())[0:3, :, :]),
                                   np.max(np.array(x_ms.cpu().detach())[0:3, :, :]),
                                   np.max(np.array(x_out.cpu().detach())[0:3, :, :]),
                                   np.max(np.array(x_gt.cpu().detach())[0:3, :, :])])) - valor_min

        x_pan = Tensor2B4B3B2(x_pan, valor_d, valor_min)
        for i in range(4):
            axs[i][0].imshow(x_pan)

        rgb_sen2 = Tensor2B4B3B2(x_ms, valor_d, valor_min)
        axs[0][1].imshow(rgb_sen2, vmin=0, vmax=1)
        rgb_pred = Tensor2B4B3B2(x_out, valor_d, valor_min)
        axs[0][2].imshow(rgb_pred, vmin=0, vmax=1)
        rgb_venus = Tensor2B4B3B2(x_gt, valor_d, valor_min)
        axs[0][3].imshow(rgb_venus, vmin=0, vmax=1)
        rgb_error = RGB_error(x_out, x_gt)
        axs[0][4].imshow(rgb_error)

        b765_sen2 = Tensor2B7B6B5(x_ms)
        axs[1][1].imshow(b765_sen2, vmin=0, vmax=1)
        b765_pred = Tensor2B7B6B5(x_out)
        axs[1][2].imshow(b765_pred, vmin=0, vmax=1)
        b765_venus = Tensor2B7B6B5(x_gt)
        axs[1][3].imshow(b765_venus, vmin=0, vmax=1)
        b765_error = np.abs(b765_venus - b765_pred)
        axs[1][4].imshow(b765_error, vmin=0, vmax=1)

        b8_sen2 = Tensor2B8(x_ms)
        axs[2][1].imshow(b8_sen2, vmin=0, vmax=1, cmap='magma')
        b8_pred = Tensor2B8(x_out)
        axs[2][2].imshow(b8_pred, vmin=0, vmax=1, cmap='magma')
        b8_venus = Tensor2B8(x_gt)
        axs[2][3].imshow(b8_venus, vmin=0, vmax=1, cmap='magma')
        b8_error = np.abs(b8_venus - b8_pred)
        axs[2][4].imshow(b8_error, vmin=0, vmax=1, cmap='magma')

        b8A_sen2 = Tensor2B8A(x_ms)
        axs[3][1].imshow(b8A_sen2, vmin=0, vmax=1, cmap='magma')
        b8A_pred = Tensor2B8A(x_out)
        axs[3][2].imshow(b8_pred, vmin=0, vmax=1, cmap='magma')
        b8A_venus = Tensor2B8A(x_gt)
        axs[3][3].imshow(b8A_venus, vmin=0, vmax=1, cmap='magma')
        b8A_error = np.abs(b8A_venus - b8A_pred)
        axs[3][4].imshow(b8A_error, vmin=0, vmax=1, cmap='magma')

    [[ax.set_yticks([]) for ax in axs_row] for axs_row in axs]
    [[ax.set_xticks([]) for ax in axs_row] for axs_row in axs]
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def floatTensor2uint8(img):
    if type(img) == np.ndarray:
        img = torch.from_numpy(img)

    img = img * 255
    img = img.type(torch.uint8)
    return img


def Tensor2B4B3B2(img, valor_d, valor_min):
    img = np.moveaxis(np.array(img[0:3, :, :].cpu().detach()), 0, -1)[:, :, ::-1]
    img = (img - valor_min) / valor_d
    return img


def RGB_error(pred, gt):
    rgb_pred = np.moveaxis(np.array(pred[0:3, :, :].cpu().detach()), 0, -1)[:, :, ::-1]
    rgb_gt = np.moveaxis(np.array(gt[0:3, :, :].cpu().detach()), 0, -1)[:, :, ::-1]
    return np.abs(rgb_pred - rgb_gt)


def Tensor2B7B6B5(img):
    img = np.moveaxis(np.array(img[4:7, :, :].cpu().detach()), 0, -1)[:, :, ::-1]
    return img


def Tensor2B8(img):
    return np.array(img[3, :, :].cpu().detach())


def Tensor2B8A(img):
    return np.array(img[7, :, :].cpu().detach())
