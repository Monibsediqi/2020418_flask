#import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from base_options import BaseOptions
from test import test, color_overlay, test_lung_regression, print_gpu_stats, remove_private_tags, load_initial_models, grad_cam_efficientnet_tb, check_xray
from flask import Flask, jsonify, request, send_from_directory, abort
import hashlib, random, datetime
from PIL import Image
from functools import wraps
import numpy as np
import flask
import io
import pydicom
import nibabel as nib
import os
import time
import matplotlib.cm as cm
import torch
from torch.autograd import Variable

app = Flask(__name__)


@app.route('/fileup',methods=['POST'])
def fileup():
    upload_dir = './input'

    if 'file' not in request.files:
        return 'File is missing', 404

    file = request.files['file']

    if file.filename == '':
        return 'File is missing(noname)', 404
    
    file.save(os.path.join(upload_dir,file.filename))

    print('file up break point')
    breakpoint()
    return 'success', 200

@app.route('/output/<file_name>', methods=['GET'])
def getresult(file_name):
    print ('['+str(datetime.datetime.now())+']'+'request : /output/'+file_name)
    output_dir = './output'
    file = os.path.join(output_dir, file_name)
    print("output / file_name break point")
    breakpoint()
    return send_from_directory(output_dir, file_name)

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    print('['+str(datetime.datetime.now())+']'+'request : /predict')
    if request.method == 'POST':
        print("request.method == 'POST' break point")

        opt = BaseOptions().parse(save=False)

        #WEB_DEPLOY = True
        WEB_DEPLOY = False
        age = None
        sex = None
        pixel_spacing = None
        if WEB_DEPLOY:
            #file upload
            upload_dir = './input'

            if 'inputfile' not in request.files:
                return 'File is missing', 404

            file = request.files['inputfile']

            if file.filename == '':
                return 'File is missing(noname)', 404
            x_spacing, y_spacing = None, None
            if request.form['age'] != '':
                try:
                    age = int(request.form['age'])
                except:
                    age = None
            if request.form['sex'] != '':
                sex = request.form['sex']
            if request.form['x_spacing'] != '':
                try:
                    x_spacing = float(request.form['x_spacing'])
                except:
                    x_spacing = None
            if request.form['y_spacing'] != '':
                try:
                    y_spacing = float(request.form['y_spacing'])
                except:
                    y_spacing = None
            #if file.pixel_spacing != '':
            #    pixel_spacing = file.pixel_spacing

            if x_spacing is not None and y_spacing is not None:
                pixel_spacing = [x_spacing, y_spacing]

            #upload file save
            input_filename = os.path.join(upload_dir, file.filename)
            file.save(input_filename)
            lungoutput_filename = hashlib.md5(str(random.getrandbits(256)).encode('utf-8')).hexdigest()+'.jpg'
            covidoutput_filename = hashlib.md5(str(random.getrandbits(256)).encode('utf-8')).hexdigest()+'.jpg'
        else:
            input_filename = request.values['input_filename']
            covidoutput_filename = request.values['covidoutput_filename']
            lungoutput_filename = covidoutput_filename + '_lung.png'
            
        #output dir
        output_dir = './output'
        
        lungnii = os.path.join(output_dir, lungoutput_filename + '.nii')
        covidnii = os.path.join(output_dir, covidoutput_filename + '.nii')
        lungpath = os.path.join(output_dir, lungoutput_filename)
        covidpath = os.path.join(output_dir, covidoutput_filename)
        print(input_filename, lungoutput_filename, covidoutput_filename)

        message = 'Success'
        success = True

        opt.test_input = input_filename
        opt.test_output = lungpath
        opt.get_lung_area = True
        opt.hn = True
        opt.output_min_max = "-1100,-500"
        opt.checkpoint_path = "./checkpoints/xray2lung.pth"
        opt.threshold = -1015
        opt.save_input = True
        opt.profnorm = True
        opt.check_xray = False
        opt.age = age
        opt.sex = sex
        opt.pixel_spacing = pixel_spacing
        try:
            print("test break point")
            breakpoint()
            lungarea, is_xray, sex, age, default_sex_age, original_width, original_height = test(opt)
        except Exception as e:
            is_xray = False
            lungarea = -2
            lungoutput_filename = ''
            covidoutput_filename = ''
            message = 'TiSepX Lung Prediction Exception: ' + str(e)
            success = False

            import sys, traceback
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            del exc_info
            #print(e)

        # lungarea = test(opt, device_num=1)

        if is_xray:
            try:
                print("test_lung_regression break point")
                breakpoint()
                lungvolume = test_lung_regression(lungnii, lungarea, sex, age)
            except Exception as e:
                is_xray = False
                lungvolume = -2
                lungoutput_filename = ''
                covidoutput_filename = ''
                message = 'TiSepX Lung Volume Prediction Error: ' + str(e)
                success = False

                import sys, traceback
                exc_info = sys.exc_info()
                traceback.print_exception(*exc_info)
                del exc_info
                #print(e)

            opt.test_input = lungnii
            opt.test_output = covidpath
            opt.get_lung_area = False
            opt.hn = False
            opt.get_covid_area = True
            #covid
            opt.checkpoint_path = "./checkpoints/lung2covid.pth"
            opt.threshold = -950  # -900# change covid threshold to -950 220902
            opt.input_min_max = "-1100,-500"
            opt.output_min_max = "-1100,-400"
            #pulmonary
            #opt.checkpoint_path = "./checkpoints/lung2vessel.pth"
            #opt.threshold = -1015
            #opt.input_min_max = "-1100,-500"
            #opt.output_min_max = "-1100,-500"
            opt.save_input = False
            opt.profnorm = False
            opt.check_xray = False
            opt.age = age
            opt.sex = sex
            opt.pixel_spacing = pixel_spacing

            try:
                covidarea, _, _, _, _, _, _ = test(opt)
                # covidarea = test(opt, device_num=1)
                color_overlay(lungnii, covidnii, covidpath, original_width, original_height)
                #os.remove(lungnii)
                #os.remove(covidnii)
                #covidarea = 0
                #eps = 1e-10
                #lung = nib.load(lungnii)
                #lung_arr = np.transpose(np.array(lung.dataobj)[..., 0, 0], axes=[1, 0])
                #lung_img = ((lung_arr - lung_arr.min()) / ((lung_arr.max() - lung_arr.min()) + eps)) * 255
                #imsave(covidpath, lung_img.astype(np.uint8))
            except Exception as e:
                covidarea = -2
                covidoutput_filename = ''
                message = 'TiSepX Covid Prediction RuntimeError: ' + str(e)
                success = False

                import sys, traceback
                exc_info = sys.exc_info()
                traceback.print_exception(*exc_info)
                del exc_info
                #print(e)                        
        else:
            covidarea = -1
            lungvolume = -2

        print_gpu_stats("before predict() empty_cache()")
        torch.cuda.empty_cache()
        print_gpu_stats("after predict() empty_cache()")
        
        return jsonify({'lungarea': lungarea, 'covidarea': covidarea, 'lungvolume': lungvolume,
                        'lungpath': lungoutput_filename, 'covidpath': covidoutput_filename,
                        'message': message, 'success': success})
    else:
        return 'Predict!'


# Limit file upload size
def limit_content_length(max_length):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            cl = request.content_length
            if cl is not None and cl > max_length:
                abort(413)
            return f(*args, **kwargs)

        return wrapper

    return decorator


# Check Flask app status from client
@app.route("/ping", methods=["GET"])
def ping():
    return "ready"


# Perform active Tb prediction
@app.route("/snuh_tb/predict", methods=["POST"])
#@limit_content_length(30 * 1024 * 1024)
def predict_snuhtb():
    print("predict_snuhtb() start")
    classIndex = 0  # Active Tb (0), Healed Tb (1)
    imageSize = 1024
    data = {"success": False}
    start_time = time.time()

    device_num = 0
    device = 'cuda:' + str(device_num)

    # Generate random word for temporary file name
    def randomword(length):
        from datetime import datetime
        import random, string
        letters = string.hexdigits
        return datetime.today().strftime("%Y%m%d") + "_" + ''.join(random.choice(letters) for i in range(length))

    try:
        model = load_initial_models()
    except Exception as e:
        print('SNUHTB Load Model RuntimeError: ', e)
        data["success"] = False
        data["message"] = 'SNUHTB Load Model RuntimeError: ' + str(e)

        import sys, traceback
        exc_info = sys.exc_info()
        traceback.print_exception(*exc_info)
        del exc_info
        #print(e)

        torch.cuda.empty_cache()
        print_gpu_stats("predict_snuhtb model init error | empty_cache()")
        return flask.jsonify(data)

    is_xray = True

    if flask.request.method == "POST":
        # Current model name
        data["model"] = "SNUH_TB 200217"

        # WEB_DEPLOY = True
        WEB_DEPLOY = False

        if WEB_DEPLOY:
            # Handling file upload error
            if 'file' not in request.files:
                data["success"] = False
                data["message"] = "File not uploaded!"
                return flask.jsonify(data)

            if request.files['file'].filename == '':
                data["success"] = False
                data["message"] = "File not uploaded!"
                return flask.jsonify(data)

            print(request.files['file'].filename)
            file = request.files['file']
        else:
            input_filename = request.values['input_filename']
            file = 'file'

        if file is not None:
            if WEB_DEPLOY:
                print(file.content_type)
                content_type = file.content_type
                ext = file.filename.lower().split(".")[-1]
            else:
                content_type = ''
                ext = input_filename.lower().split(".")[-1]

            try:
                # Handling uploaded image and convert to 1024x1024x3 numpy array.
                # If uploaded content is png or jpeg :
                data['pixel_spacing'] = [1, 1]
                data['original_input_shape'] = [1024, 1024]
                if content_type in ["image/png", "image/jpeg", "image/bmp", "image/gif", "image/tif"] or \
                        ext in ["png", "jpeg", "jpg", "bmp", "gif", "tif"]:
                    if WEB_DEPLOY:
                        imageBytes = flask.request.files["file"].read()
                        imgobj = Image.open(io.BytesIO(imageBytes))
                    else:
                        imgobj = Image.open(input_filename)

                    imgobjmode = imgobj.mode

                    # Convert image to grayscale image
                    if imgobj.mode in ["RGB", "RGBA"]:
                        imgobjmode = "L"
                        image = np.array(imgobj.convert("L"))
                    elif imgobj.mode in ["L"]:
                        imgobjmode = "L"
                        image = np.array(imgobj.convert("L"))
                    else:
                        # Preserve 16 bit color image
                        image = np.array(imgobj.convert("I"))

                    if image.shape[0] is None or image.shape[1] is None:
                        image = np.array(imgobj.convert("L"))
                    if image.shape[0] is None or image.shape[1] is None:
                        data["success"] = False
                        data["message"] = "Corrupted File"
                        return flask.jsonify(data)
                    image = Image.fromarray(image, mode=imgobjmode)

                # If uploaded content is dicom file :
                elif content_type in ["application/dicom"] or ext in ["dcm", "dc3", "dicom", "nii"]:
                    if WEB_DEPLOY:
                        tempDCMPath = "tempdcm/" + randomword(20)
                        file.save(tempDCMPath)
                    else:
                        tempDCMPath = input_filename

                    if ext in ["dcm", "dc3", "dicom"]:
                        dataset = pydicom.dcmread(tempDCMPath, force=True)
                        dataset = remove_private_tags(dataset, tempDCMPath)
                        new_array = dataset.pixel_array

                        try:
                            assert dataset.ImagerPixelSpacing[0] == dataset.ImagerPixelSpacing[1]
                            pixel_spacing = dataset.ImagerPixelSpacing[:2]
                        except:
                            try:
                                assert dataset.PixelSpacing[0] == dataset.PixelSpacing[1]
                                pixel_spacing = dataset.PixelSpacing[:2]
                            except:
                                pixel_spacing = [1, 1]
                        data['pixel_spacing'] = pixel_spacing
                        data['original_input_shape'] = new_array.shape

                    elif ext in ["nii"]:
                        dataset = nib.load(tempDCMPath)
                        nii_shape = len(np.array(dataset.dataobj).shape)

                        if nii_shape == 2:
                            new_array = np.transpose(np.array(dataset.dataobj), axes=[1, 0])
                        elif nii_shape == 3:
                            new_array = np.transpose(np.array(dataset.dataobj), axes=[2, 1, 0])[0, :, :]
                        elif nii_shape == 4:
                            new_array = np.transpose(np.array(dataset.dataobj), axes=[3, 2, 1, 0])[0, 0, :, :]

                        header = dataset.header
                        data['pixel_spacing'] = header.get_zooms()
                        data['original_input_shape'] = np.transpose(new_array, axes=[1, 0]).shape

                    mean = np.mean(new_array)
                    std = np.std(new_array)

                    new_array[new_array < mean - std * 2] = mean - std * 2
                    np05 = np.nanpercentile(new_array, 0)
                    np95 = np.nanpercentile(new_array, 99)
            
                    normalized_array = (new_array - np05) * 65535.0 / (np95 - np05)
                    normalized_array[normalized_array > 1 * 65535] = 65535
                    normalized_array[normalized_array < 0 * 65535] = 0 * 65535

                    if ext in ["dcm", "dc3", "dicom"] and str(dataset[0x28, 0x04].value) == "MONOCHROME1":
                        normalized_array = 65535 - normalized_array

                    imageArr = normalized_array.astype('uint16')
                    imgobjmode = "I"

                    array_buffer = imageArr.tobytes()
                    image = Image.new("I", normalized_array.T.shape)
                    image.frombytes(array_buffer, 'raw', "I;16")

                # os.remove(tempDCMPath)
                # Not supported file type
                else:
                    data["success"] = False
                    data["message"] = "Unidentifiable file"
                    return flask.jsonify(data)

                # Convert image resolution:
                # Target : 1024x1024x3

                image.thumbnail((imageSize, imageSize), Image.ANTIALIAS)
                image = np.array(image)

                is_xray = check_xray(image, device)

                if is_xray:
                    # If image is smaller than 1024x1024:
                    if image.shape[0] != 1024 or image.shape[1] != 1024:
                        mode = "L"
                        if imgobjmode == "I":
                            mode = "I"

                        image = Image.fromarray(image)
                        x, y = image.size
                        newim = Image.new(mode, (imageSize, imageSize), (0))
                        newim.paste(image, (int((imageSize - x) / 2), int((imageSize - y) / 2)))
                        image = np.array(newim)
                    # If image is 16 bit image :
                    if imgobjmode == "I" or np.max(image) > 255:
                        rescale = 65535.0
                        black8bitimg = np.clip(image / 255.0, 0, 255).astype(np.uint8)
                    # Else image is 8 bit image :
                    else:
                        rescale = 255.0
                        black8bitimg = np.clip(image, 0, 255).astype(np.uint8)

                    # Convert gray scale image to RGB color image
                    # The neural network model accepts 1024x1024x3 as an input.
                    # Rescale numpy array to float32 value between 0 and 1.
                    modelInputImage = np.stack((image,) * 3, axis=-1) / rescale
                    modelInputImage = np.expand_dims(modelInputImage, axis=0)
                    modelInputImage = np.vstack([modelInputImage])
                    modelInputImage = np.transpose(modelInputImage, axes=[0, 3, 1, 2])
                    modelInputImage = Variable(torch.from_numpy(modelInputImage.astype(np.float32))).to(device)
                    # with graph.as_default():

                    activation = {}

                    def get_activation(name):
                        def hook(model, input, prediction):
                            activation[name] = prediction

                        return hook

                    model._softmax.register_forward_hook(get_activation('_softmax'))
                    model._last_swish.register_forward_hook(get_activation('_last_swish'))

                    model.to(device)
                    model.eval()
                    # with torch.no_grad():
                    # Perform prediction
                    prediction = model(modelInputImage)
                    print(prediction)
                    data["prediction"] = prediction[0][0].cpu().detach().numpy() * 100

                    # Generate "decision" string to display on client.
                    decision = 'N/A'
                    if prediction[0][0] > 0.8:
                        decision = "Definite active Tb"
                    elif prediction[0][0] > 0.5084:
                        decision = "Probable active Tb"
                    elif prediction[0][0] > 0.25:
                        decision = "Probable healed Tb"
                    else:
                        decision = "Definite healed Tb"

                    data['decision'] = decision

                    # Generate CAM
                    cam = grad_cam_efficientnet_tb(img=modelInputImage, model=model, class_index=classIndex,
                                                   activation_layer=activation, imageSize=imageSize)

                    # Generate color image from CAM array
                    cam = np.clip(cam / cam.max(), 0, 1.0)
                    grey = black8bitimg
                    grey = np.clip(grey / np.nanmax(grey) + 0.2, 0, 1.0)
                    grey = grey * 0.5 * 1 * prediction[0][classIndex].cpu().detach().numpy()
                    grey = np.clip(grey, 0, 0.8)
                    cam = np.clip(cam / cam.max() * prediction[0][classIndex].cpu().detach().numpy(), 0, 1)
                    rgb = cm.jet(cam * 1.0)[..., :3]
                    rgba = np.dstack((rgb, grey)) * 255.0
                    rgba = rgba.astype('uint8')

                    background = Image.fromarray(black8bitimg)
                    foreground = Image.fromarray(rgba)
                    foreground.putalpha(32)
                    overlay = Image.new("RGBA", background.size)
                    overlay = Image.alpha_composite(overlay, background.convert('RGBA'))
                    overlay = Image.alpha_composite(overlay, foreground)

                    # Save result image on web server to serve client
                    random = randomword(20)
                    # colorFilename = random+"-color.png"
                    originalFilename = random + "-original.png"
                    overlayFilename = random + "-overlay.png"
                    # exportColorPath = "output_tb/" +colorFilename
                    exportOriginalPath = "output_tb/" + originalFilename
                    exportOverlayPath = "output_tb/" + overlayFilename
                    # Image.fromarray(rgba).save(exportColorPath ,format="png")
                    Image.fromarray(black8bitimg).save(exportOriginalPath, format="png")
                    overlay.save(exportOverlayPath, format="png")
                    thumb_size = 200, 200
                    thumb_img = Image.fromarray(black8bitimg)
                    thumb_img.thumbnail(thumb_size, Image.ANTIALIAS)
                    thumb_img.save(exportOriginalPath[:-4] + "_thumbnail" + ".png", format="png")
                    overlay.thumbnail(thumb_size, Image.ANTIALIAS)
                    overlay.save(exportOverlayPath[:-4] + "_thumbnail" + ".png", format="png")
                    #overlay.save(exportOverlayPath, format="png")

                    data["color"] = "output_tb/" + overlayFilename
                    data["original"] = "output_tb/" + originalFilename

                    tb_pixel_count = np.sum(np.where(cam >= 0.16, 1, 0).flatten())
                    data["tb_pixel_count"] = tb_pixel_count

                    data["success"] = True

                    modelInputImage = modelInputImage.cpu()
                    activation = {}
                    prediction = prediction.cpu().detach()
                else:
                    data["prediction"] = -1
                    data["success"] = False
            except Exception as e:
                # Handling all kinds of error
                data["success"] = False
                data["message"] = "Unknown error"

                import sys, traceback
                exc_info = sys.exc_info()
                traceback.print_exception(*exc_info)
                del exc_info
                print(e)
    try:
        model.cpu()
    except Exception as e:
        import sys, traceback
        exc_info = sys.exc_info()
        traceback.print_exception(*exc_info)
        del exc_info
        print(e)
    # Calculating elapsed time during calculation.
    elapsed_time = time.time() - start_time
    data['elapsed_time'] = elapsed_time

    # Prevent memory leak, dealloc variables
    import gc
    gc.collect()
    
    print_gpu_stats("before predict_snuhtb() empty_cache()")
    torch.cuda.empty_cache()
    print_gpu_stats("after predict_snuhtb() empty_cache()")

    if data['success']:
        #######################
        # Predict TiSepX lung #
        #######################
        opt = BaseOptions().parse(save=False)
        opt.test_input = input_filename
        opt.test_output = input_filename + 'temp'
        opt.get_lung_area = True
        opt.hn = True
        opt.output_min_max = "-1100,-500"
        opt.checkpoint_path = "./checkpoints/xray2lung.pth"
        opt.threshold = -1015
        opt.save_input = False
        opt.profnorm = True
        opt.check_xray = True
        opt.age = None
        opt.sex = None
        opt.pixel_spacing = None

        try:
            lungarea, _, _, _, _, _, _ = test(opt)
            os.remove(input_filename + 'temp' + '.nii')        
        except Exception as e:
            is_xray = False
            lungarea = -2
            lungoutput_filename = ''
            covidoutput_filename = ''
            message = 'TiSepX Lung Prediction Exception: ' + str(e)
            success = False

            import sys, traceback
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            del exc_info
            # print(e)
        #######################
        # Predict TiSepX lung #
        #######################

        #######################
        # Calculate TB Extent #
        #######################
        try:
            original_height = data['original_input_shape'][0]
            original_width = data['original_input_shape'][1]
            ratio = float(1024) / max(original_width, original_height)

            pixel_size_resize_w = data['pixel_spacing'][0] / ratio
            pixel_size_resize_h = data['pixel_spacing'][1] / ratio
            tb_area = data['tb_pixel_count'] * pixel_size_resize_w * pixel_size_resize_h / 100

            tb_extent = tb_area / lungarea * 100
            data['tb_extent'] = tb_extent      
        except Exception as e:
            import sys, traceback
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            del exc_info
        
        #######################
        # Calculate TB Extent #
        #######################

    data['tb_pixel_count'] = None
    data['pixel_spacing'] = None
    data['original_input_shape'] = None

    gc.collect()
    
    print_gpu_stats("before predict() empty_cache()")
    torch.cuda.empty_cache()
    print_gpu_stats("after predict() empty_cache()")

    return flask.jsonify(data)


def resize_and_pad(img, size, padColor=0):
    import cv2

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img


def npz_batch_from_dcm_and_tisepx(dcm_file, tisepx_file, resolution=(1024, 1024), precision=np.float64, channel_first=False):
    import cv2
    import nibabel as nib

    # load tisepx
    img = nib.load(tisepx_file+".nii")
    pixel_array = img.get_fdata()
    pixel_array = np.squeeze(pixel_array)
    pixel_array = np.expand_dims(pixel_array, axis=2)
    pixel_array = cv2.rotate(pixel_array, cv2.ROTATE_90_CLOCKWISE)
    pixel_array = cv2.flip(pixel_array, 1)

    normalized_array = (lambda x:(x-x.min())/(x.max()-x.min()))(pixel_array.astype(np.float64))
    resized = resize_and_pad(normalized_array, (2048,2048))
    lung = np.clip(resized, 0.0, 1.0).astype(np.float64)

    resized = resize_and_pad(dcm_file, (2048,2048))
    cxr = np.clip(resized, 0.0, 1.0).astype(np.float64)
    lung_binary_array = np.where(lung > 0.05, 1, 0)
    cxr_segmented = cxr * lung_binary_array

    cxr_segmented = cxr_segmented.astype(precision)
    cxr_segmented = np.stack((cxr_segmented,) * 3, axis=-1)
    cxr_segmented = np.clip(cxr_segmented, 0, 1).astype(precision)
    cxr_segmented = cv2.resize(cxr_segmented, dsize=resolution, interpolation=cv2.INTER_LINEAR)
    if channel_first:
        cxr_segmented = np.moveaxis(cxr_segmented, -1, 0)

    cxr_segmented = cxr_segmented[np.newaxis, ...]
    return cxr_segmented


def read_dcm_to_img_arr(dcm_file, output_size_rectangle=(1024,1024), precision=np.float64, channel_first=False):
    import cv2

    resized = cv2.resize(dcm_file, dsize=output_size_rectangle, interpolation=cv2.INTER_LINEAR)
    resized = np.stack((resized,) * 3, axis=-1)
    resized = np.clip(resized, 0, 1).astype(precision)
    if channel_first:
        resized = np.moveaxis(resized, -1, 0)

    return resized


def pytorch_inference_with_grad_cam(dcm_file, tisepx_file, pytorch_model_path, output_grad_cam_png_path, output_ori_path, device):
    import torch, cv2
    from pytorch_grad_cam import GradCAM, EigenGradCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from PIL import Image
    
    #torch.set_default_dtype(torch.float64)

    batch = npz_batch_from_dcm_and_tisepx(dcm_file=dcm_file,tisepx_file=tisepx_file,resolution=(1024,1024),
                                                    precision=np.float32,channel_first=True)#np.float64,channel_first=True)
    pytorch_model = torch.load(pytorch_model_path)
    pytorch_model.to(device)
    pytorch_model.eval()
    #pytorch_model.double()
    modelInputImage = torch.tensor(batch).to(device)
    #modelInputImage = torch.tensor(batch)
    y = pytorch_model(modelInputImage).detach().cpu().numpy()
    prediction = y[0][1]
    print("PYTORCH positive prediction : %.3f" % float(prediction))

    for name, module in pytorch_model.named_modules():
        if name == 'Mul_StatefulPartitionedCall/model/top_activation/mul:0':
            break

    #cam = GradCAM(model=pytorch_model, target_layers=[module
    cam = GradCAMPlusPlus(model=pytorch_model, target_layers=[module])
    #cam = EigenGradCAM(model=pytorch_model, target_layers=[module])
    targets = [ClassifierOutputTarget(1)]
    #heatmap = cam(input_tensor=torch.tensor(batch), targets=targets)
    heatmap = cam(input_tensor=modelInputImage, targets=targets, aug_smooth = False, eigen_smooth = False)
    #True False
    heatmap = np.moveaxis(heatmap,0,-1)
    #import matplotlib.pyplot as plt
    #plt.imshow(np.where(heatmap >= 0.16, 1, 0))
    #plt.imshow(batch[0][0])
    #plt.show()
    
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap2 = np.clip(heatmap / heatmap.max() * prediction, 0, 1)
    tb_pixel_count = np.sum(np.where(heatmap2 >= 0.16, 1, 0).flatten())
    #image = read_dcm_to_img_arr(dcm_file=dcm_file)
    #lung only
    image = np.stack((batch[0][0],) * 3, axis=-1)
    image = np.clip(image, 0, 1).astype(np.float64)
    
    #image = np.moveaxis(batch, 0, -1)
    
    heatmap = np.clip((1-heatmap) * 255.0, 0, 255).astype(np.uint8)
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    cam = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    alpha = prediction / 3
    output = cv2.addWeighted(image, 1 - alpha, cam, alpha, 0)
    Image.fromarray(image).save(output_ori_path, format="png")
    Image.fromarray(output).save(output_grad_cam_png_path, format="png")
    #, format="png"
    modelInputImage.cpu()
    pytorch_model.cpu()
    return prediction * 100, int(tb_pixel_count)

# Perform active Tb prediction 20230130
@app.route("/snuh_tb/predict2", methods=["POST"])
#@limit_content_length(30 * 1024 * 1024)
def predict_snuhtb2():
    data = {"success": False}
    start_time = time.time()

    device_num = 0
    device = 'cuda:' + str(device_num)

    if flask.request.method == "POST":
        # Current model name
        data["model"] = "SNUH_TB 202301"

        # WEB_DEPLOY = True
        WEB_DEPLOY = False

        if WEB_DEPLOY:
            # Handling file upload error
            if 'file' not in request.files:
                data["success"] = False
                data["message"] = "File not uploaded!"
                return flask.jsonify(data)

            if request.files['file'].filename == '':
                data["success"] = False
                data["message"] = "File not uploaded!"
                return flask.jsonify(data)

            print(request.files['file'].filename)
            file = request.files['file']
        else:
            input_filename = request.values['input_filename']
            file = 'file'

    #######################
    # Predict TiSepX lung #
    #######################
    opt = BaseOptions().parse(save=False)
    opt.test_input = input_filename
    opt.test_output = input_filename + 'temp'
    opt.get_lung_area = True
    opt.output_min_max = "-1100,-500"
    opt.checkpoint_path = "./checkpoints/xray2lung.pth"
    opt.threshold = -1015
    opt.save_input = False
    opt.profnorm = True
    opt.check_xray = False
    opt.age = None
    opt.sex = None
    opt.pixel_spacing = None
    opt.hn = True

    #### STARTING POINT ####
    try:
        lungarea, _, _, _, _, _, _ = test(opt)
        # os.remove(input_filename + 'temp' + '.nii')
        data['success'] = True
    except Exception as e:
        is_xray = False
        lungarea = -2
        lungoutput_filename = ''
        covidoutput_filename = ''
        message = 'TiSepX Lung Prediction Exception: ' + str(e)
        success = False

        import sys, traceback
        exc_info = sys.exc_info()
        traceback.print_exception(*exc_info)
        del exc_info
        # print(e)

    #######################
    # Predict TiSepX lung #
    #######################

    #######################
    ##### Predict TB ######
    #######################

    # Generate random word for temporary file name
    def randomword(length):
        from datetime import datetime
        import random, string
        letters = string.hexdigits
        return datetime.today().strftime("%Y%m%d") + "_" + ''.join(random.choice(letters) for i in range(length))

    opt.pytorch_model_path = "./checkpoints/tiseptb2.pth"
    opt.tisepx_lung = input_filename + 'temp'
    random = randomword(20)
    opt.tb_output = './output_tb/' + random + '-overlay.png'
    opt.ori_output = './output_tb/' + random + '-original.png'

    data["color"] = opt.tb_output
    data["original"] = opt.ori_output

    if WEB_DEPLOY:
        print(file.content_type)
        content_type = file.content_type
        ext = file.filename.lower().split(".")[-1]
    else:
        content_type = ''
        ext = input_filename.lower().split(".")[-1]

    try:
        data['pixel_spacing'] = [1, 1]
        data['original_input_shape'] = [2048, 2048]
        if content_type in ["image/png", "image/jpeg", "image/bmp", "image/gif", "image/tif"] or \
                ext in ["png", "jpeg", "jpg", "bmp", "gif", "tif"]:
            if WEB_DEPLOY:
                imageBytes = flask.request.files["file"].read()
                imgobj = Image.open(io.BytesIO(imageBytes))
            else:
                imgobj = Image.open(input_filename)

            imgobjmode = imgobj.mode

            # Convert image to grayscale image
            if imgobj.mode in ["RGB", "RGBA"]:
                imgobjmode = "L"
                image = np.array(imgobj.convert("L"))
            elif imgobj.mode in ["L"]:
                imgobjmode = "L"
                image = np.array(imgobj.convert("L"))
            else:
                # Preserve 16 bit color image
                image = np.array(imgobj.convert("I"))

            if image.shape[0] is None or image.shape[1] is None:
                image = np.array(imgobj.convert("L"))
            if image.shape[0] is None or image.shape[1] is None:
                data["success"] = False
                data["message"] = "Corrupted File"
                return flask.jsonify(data)
            # image = Image.fromarray(image, mode=imgobjmode)
            f = (image - np.min(image)) / (np.max(image) - np.min(image))
            import matplotlib.pyplot as plt
            plt.imshow(f)

        # If uploaded content is dicom file :
        elif content_type in ["application/dicom"] or ext in ["dcm", "dc3", "dicom", "nii"]:
            if WEB_DEPLOY:
                tempDCMPath = "tempdcm/" + randomword(20)
                file.save(tempDCMPath)
            else:
                tempDCMPath = input_filename

            if ext in ["dcm", "dc3", "dicom"]:
                dataset = pydicom.dcmread(tempDCMPath, force=True)
                # dataset = remove_private_tags(dataset, tempDCMPath)
                new_array = dataset.pixel_array

                try:
                    assert dataset.ImagerPixelSpacing[0] == dataset.ImagerPixelSpacing[1]
                    pixel_spacing = dataset.ImagerPixelSpacing[:2]
                except:
                    try:
                        assert dataset.PixelSpacing[0] == dataset.PixelSpacing[1]
                        pixel_spacing = dataset.PixelSpacing[:2]
                    except:
                        pixel_spacing = [1, 1]
                data['pixel_spacing'] = pixel_spacing
                data['original_input_shape'] = new_array.shape

            elif ext in ["nii"]:
                dataset = nib.load(tempDCMPath)
                nii_shape = len(np.array(dataset.dataobj).shape)

                if nii_shape == 2:
                    new_array = np.transpose(np.array(dataset.dataobj), axes=[1, 0])
                elif nii_shape == 3:
                    new_array = np.transpose(np.array(dataset.dataobj), axes=[2, 1, 0])[0, :, :]
                elif nii_shape == 4:
                    new_array = np.transpose(np.array(dataset.dataobj), axes=[3, 2, 1, 0])[0, 0, :, :]

                header = dataset.header
                data['pixel_spacing'] = [float(h) for h in header.get_zooms()]
                data['original_input_shape'] = np.transpose(new_array, axes=[1, 0]).shape

            mean = np.mean(new_array)
            std = np.std(new_array)

            new_array[new_array < mean - std * 2] = mean - std * 2
            np05 = np.nanpercentile(new_array, 0)
            np95 = np.nanpercentile(new_array, 99)

            # normalized_array = (new_array - np05) * 65535.0 / (np95 - np05)
            # normalized_array[normalized_array > 1 * 65535] = 65535
            # normalized_array[normalized_array < 0 * 65535] = 0 * 65535
            normalized_array = (new_array - np05) / (np95 - np05)
            normalized_array = np.clip(normalized_array, 0, 1)

            if ext in ["dcm", "dc3", "dicom"] and str(dataset[0x28, 0x04].value) == "MONOCHROME1":
                # normalized_array = 65535 - normalized_array
                normalized_array = 1 - normalized_array

            s = max(normalized_array.shape[0:2])
            f = np.zeros((s, s), np.float64)
            ax, ay = (s - normalized_array.shape[1]) // 2, (s - normalized_array.shape[0]) // 2
            f[ay:normalized_array.shape[0] + ay, ax:ax + normalized_array.shape[1]] = normalized_array

        else:
            data["success"] = False
            data["message"] = "Unidentifiable file"
            return flask.jsonify(data)
        # data['tb_pixel_count']
        data["prediction"], data['tb_pixel_count'] = pytorch_inference_with_grad_cam(dcm_file=f,
                                                                                 tisepx_file=opt.tisepx_lung,
                                                                                 pytorch_model_path=opt.pytorch_model_path,
                                                                                 output_grad_cam_png_path=opt.tb_output,
                                                                                 output_ori_path=opt.ori_output, device=device)
        # Generate "decision" string to display on client.
        # decision = 'N/A'
        if data["prediction"] > 80:
            decision = "Definite active Tb"
        elif data["prediction"] > 50.84:
            decision = "Probable active Tb"
        elif data["prediction"] > 25:
            decision = "Probable healed Tb"
        else:
            decision = "Definite healed Tb"

        data['decision'] = decision

        #######################
        # Calculate TB Extent #
        #######################
        original_height = data['original_input_shape'][0]
        original_width = data['original_input_shape'][1]
        ratio = float(1024) / max(original_width, original_height)

        pixel_size_resize_w = data['pixel_spacing'][0] / ratio
        pixel_size_resize_h = data['pixel_spacing'][1] / ratio
        tb_area = data['tb_pixel_count'] * pixel_size_resize_w * pixel_size_resize_h / 100
        
        tb_extent = tb_area / lungarea * 100
        data['tb_extent'] = tb_extent
        data['lungarea'] = lungarea
        data['tbarea'] = tb_area
        #######################
        # Calculate TB Extent #
        #######################

    except Exception as e:
        # Handling all kinds of error
        data["success"] = False
        data["message"] = "Unknown error"

        import sys, traceback
        exc_info = sys.exc_info()
        traceback.print_exception(*exc_info)
        del exc_info
        print(e)

    #######################
    ##### Predict TB ######
    #######################

    os.remove(opt.tisepx_lung +".nii")

    # Calculating elapsed time during calculation.
    elapsed_time = time.time() - start_time
    data['elapsed_time'] = elapsed_time

    # Prevent memory leak, dealloc variables
    import gc
    gc.collect()
    print("break point 1")
    breakpoint()
    print_gpu_stats("before predict_snuhtb() empty_cache()")
    torch.cuda.empty_cache()
    print_gpu_stats("after predict_snuhtb() empty_cache()")

    return flask.jsonify(data)

if __name__ == "__main__":

    print("Using main? ")
    app.debug = True
    from flask_cors import CORS
    from OpenSSL import SSL

    # cross origin setting
    # 백엔드/프론트엔드가 같은 서버라면 필요 없는 코드
    #cors = CORS(app, resources={r"/*": {"origin": "http://106.254.254.173:5001"}})
    #app.run(host='0.0.0.0', port=45000, debug=True)

    cors = CORS(app, resources={r"/*": {"origin": "app.radiologist.app"}})
    app.run(host='127.0.0.1', port=5000, debug=True)

    # 도커 컨테이너와 통신하기 위해 ip를 0.0.0.0으로 설정
    # app.run(host='127.0.0.1', port=5000, debug=True, ssl_context=('fullchain.pem', 'privkey.pem'))

