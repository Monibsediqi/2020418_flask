import requests

if __name__ == '__main__':
    #payload = {'input_filename': './input_tb/41.png',
    payload = {'input_filename': './input_tb/10091724-X00001.dcm',
    #15505521-X00001
    #15554903-X00001
    #10091724-X00001
    #10206920-X00001
    #10241877-X00001
    #11075509-X00001
    #11718806-X00001
    #12368499-X00001
    #12691795-X00001
    #13112187-X00001
    #13301211-X00001
    #13508245-X00001
    #14197299-X00001
    #14699889-X00001
    #15143271-X00001
    #payload = {'input_filename': './input_tb/5_Random number_262.nii',
    #payload = {'input_filename': './input_tb/001.dcm',
    # payload = {'input_filename': './input_tb/250109.dc3',
    # payload = {'input_filename': './input_tb/700000089_2016-11-26_input.png',
    # payload = {'input_filename': './input_tb/46042846.DCM',
    # payload = {'input_filename': './input_tb/MCUCXR_0004_0.png',
    # payload = {'input_filename': './input_tb/MCUCXR_0013_0.png',
    # payload = {'input_filename': './input_tb/KICC_02007_20200329.DCM.jpg',
    # payload = {'input_filename': './input_tb/AP00001.dcm',
    # payload = {'input_filename': './input_tb/ILD.DCM',
    # payload = {'input_filename': './input_tb/2020.04.10_CHEST AP00001bmp.tif',
    # payload = {'input_filename': './input_tb/ct1.dcm',
    # payload = {'input_filename': './input_tb/ct2.dcm',
    # payload = {'input_filename': './input_tb/ct3.nii',
               }

    resp = requests.post("http://localhost:5000/snuh_tb/predict2", data=payload)

    print(resp.json())