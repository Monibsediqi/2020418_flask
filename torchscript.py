import datetime
import gc
import os
import nibabel as nib
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import pathlib
from collections import OrderedDict
from test import create_model
from base_options import BaseOptions


temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

checkpoint_file = 'checkpoints\\xray2lung.pth'
scripted_model_path = 'checkpoints\\xray2lung.mipx'


def normalize(image):
    """
    Min-Max Normalization
    :param image: 3D image
    """
    img_3d = image
    mean_val = torch.mean(img_3d)
    std_val = torch.std(img_3d)

    rescale_slope = std_val
    rescale_intercept = mean_val

    return (img_3d - rescale_intercept) / rescale_slope


def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    return new_state_dict


def build_model(checkpoint):

    input_filename = "./input_tb/10091724-X00001.dcm"
    opt = BaseOptions().parse(save=False)
    opt.test_input = input_filename
    opt.test_output = input_filename + 'temp'
    opt.get_lung_area = True
    opt.output_min_max = "-1100,-500"
    opt.checkpoint_path = checkpoint
    opt.threshold = -1015
    opt.save_input = False
    opt.profnorm = True
    opt.check_xray = False
    opt.age = None
    opt.sex = None
    opt.pixel_spacing = None
    opt.hn = True
    opt.use_gpu = 0
    device ='cuda:' + str(0)
    opt.cuda0 = device
    opt.cuda1 = device

    model = create_model(opt)
    return model


if __name__ == "__main__":

    model = build_model(checkpoint_file)
    model.eval()
    model.float()
    dumy_data = torch.rand(1, 1, 2048, 2048).type(torch.cuda.FloatTensor)
    dumy_inst = torch.rand(1).type(torch.cuda.FloatTensor)
    dumy_img = torch.rand(1).type(torch.cuda.FloatTensor)


    traced_script_module = torch.jit.trace(model, (dumy_data, dumy_inst), check_trace=False)
    # traced_script_module = torch.jit.trace_module(model, (dumy_data,  dumy_inst), check_trace=False)



    print('model', model)


    # model_G = build_model_G(args)
    # new_state_dict = remove_data_parallel(checkpoint['model_G'])

    # model_G.load_state_dict(new_state_dict)
    # model_G = model_G.cuda()
    # model_G_half = model_G.half()
    # print("model device: ", model_G.device)

    # # # # Method 1: Tracing
    # example_input = torch.rand(1, 1, 96, 96, 96, dtype=torch.float16).cuda()
    # torch.jit.trace(model_G,  example_input, check_trace=False).save(scripted_model_path)
    print("Tracing done")

    # -------------------- Testing Torchscript Model --------------------
    # loading torchscript model
    # scripted_model_path = 'traced_model_cuda.pt'
    # torchscript_model = torch.jit.load(scripted_model_path)
    #
    # checkpoint = torch.load(checkpoint_file)
    # args = checkpoint['args']
    # args.device = 'cuda'
    # args.use_checkpoint = False
    # # args.out_channels = 8
    # print("args: ", args)
    # args.val_A_data_path = "D:\\2023\\001_Projects\\P001_Segmentation\\P001_003_DT_Dataset\\deepcatch_data\\data"
    # args.val_B_data_path = "D:\\2023\\001_Projects\\P001_Segmentation\\P001_003_DT_Dataset\\deepcatch_data\\gt_nii"
    # args.val_data_save_path = 'D:\\2023\\001_Projects\\P001_Segmentation\\P001_005_RT_Result\\DeepCatch_Result_Sample'
    #
    # main(args, torchscript_model)

    # ---------------------- Testing with a single downsized image ----------------------
    # # loading torchscript model
    import numpy as np
    # scripted_model_path = 'scripted_models/traced_model_cpu.pt'
    # torchscript_model = torch.jit.load(scripted_model_path)
    #
    # # # loading a single image
    # image_path = "F:\\2023\\AI\\sample_data\\image_a\\Case24_resized_96_v1.nii"
    # mask_path = "F:\\2023\\AI\\sample_data\\image_b\\Case24_resized_96_v2.nii"
    # image = nib.load(image_path).get_fdata()
    # image = np.array(image, dtype=np.float32)
    # print(f"before norm: image min-max: {np.min(image), np.max(image)}")
    # print("image numpy: ", image.shape)
    # image = torch.from_numpy(image)
    # image = image.squeeze()
    # print("image torch: ", image.shape)
    # image = normalize(image) # min-max normalization
    # print(f"after norm: image min-max: {torch.min(image), torch.max(image)}" )
    # image = image.unsqueeze(0).unsqueeze(0)
    # image = image.cuda()
    # image = image.type(torch.FloatTensor)
    # print("image: ", image.shape)
    # fake_image = torchscript_model(image)
    # print("fake_image: ", fake_image.shape)
    # fake_image = F.softmax(fake_image, dim=1)
    # print("fake_image: ", fake_image.shape)
    # fake_image = torch.argmax(fake_image, dim=1)
    # print("fake_image: ", fake_image.shape)
    # fake_image = fake_image.squeeze(1).squeeze(0)
    # fake_image = fake_image.cpu().detach().numpy()
    # real_B = nib.load(image_path).get_fdata()
    # final_fake_image = fake_image.astype(str(real_B.dtype))
    #
    # nifti_file = nib.Nifti1Image(final_fake_image, affine=None)
    # nib.save(nifti_file, "F:\\2023\\AI\\sample_data\\image_b\\[fake]_Case24_resized_96_v1.nii")
    # print("Saved fake B image to: ", "F:\\2023\\AI\\sample_data\\image_b\\[fake]_Case24_resized_96_v1.nii")







