import requests

if __name__ == '__main__':
    #payload = {'input_filename': './input/NM-R001.dcm',
    #'covidoutput_filename': 'NM-R001.dcm.png',
    payload = {'input_filename': './input/JB0006_CXR_0base_201229.dcm',
    'covidoutput_filename': 'JB0006_CXR_0base_201229_covid.dcm.png',
    #payload = {'input_filename': './input/5_Random number_262.dcm',
    #payload = {'input_filename': './input/2020.04.10_CHEST AP00001_2.dcm',
    # payload = {'input_filename': './input/2020.04.10_CHEST AP00001.DICOM',
    # payload = {'input_filename': './input/2020.04.10_CHEST AP00001bmp.bmp',
    # payload = {'input_filename': './input/46042846.DCM',
    # payload = {'input_filename': './input/46042846.jpg',
    # payload = {'input_filename': './input/41.png',
    # payload = {'input_filename': './input/ILD.DCM',
    # payload = {'input_filename': './input/ILD.jpg',
    # payload = {'input_filename': './input/ILD.png',
    # payload = {'input_filename': './input_tb/ct1.dcm',
    # payload = {'input_filename': './input_tb/ct2.dcm',
    # payload = {'input_filename': './input_tb/ct3.nii',
    #            'covidoutput_filename': '5_Random number_262_covid.dcm.png',
    #           'covidoutput_filename': '2020.04.10_CHEST AP00001_lung.png',
    #              'covidoutput_filename': '2020.04.10_CHEST AP00001_DICOMlung.png',
    #            'covidoutput_filename': '2020.04.10_CHEST AP00001.png'
    #            'covidoutput_filename': '2020.04.10_CHEST AP00001bmp.png'
    #            'covidoutput_filename': '46042846.png'
    #            'covidoutput_filename': '46042846jpg.png'
    #            'covidoutput_filename': '41.png'
    #            'covidoutput_filename': 'ILD.png'
    #            'covidoutput_filename': 'ct1.png'
    #            'covidoutput_filename': 'ct2.png'
    #            'covidoutput_filename': 'ct3.png'
               }

    resp = requests.post("http://localhost:5000/predict", data=payload)

    print("resp.json val: ", resp.json())
