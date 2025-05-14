from pathlib import Path
from tempfile import TemporaryDirectory
import urllib.request
import os
import urllib.request
import zipfile
import subprocess
import datetime

MODEL_INFO = {'latest':{'model':'DeepLIIF', # cli.py train looks for subfolder "train" under dataroot
                        'dir_input_train':['Datasets/Sample_Dataset'],
                        'dir_input_testpy':['Datasets/Sample_Dataset'],
                        'dir_input_inference':['Datasets/Sample_Dataset/test_cli'],
                        'dir_model':['../checkpoints/DeepLIIF_Latest_Model'],
                        'modalities_no': [4],
                        'seg_gen':[True],
                        'tile_size':512,
                        'baseline':[#'tests-image-quality-baseline-latest',
                                    #'7a9c30c', # adding return_seg_intermediate
                                    'fa11236' # statistics calculation fix after adding return_seg_intermediate
                                    ],
                        'dir_output_baseline_testpy':['Datasets_baseline/Sample_Dataset/test'],
                        'dir_output_baseline_inference':['Datasets_baseline/Sample_Dataset/test_cli'],
                        'suffix':{'testpy':{'intermediate_mod':['_B_1.png','_B_2.png','_B_3.png','_B_4.png'],'seg':['_B_5.png']},
                                  'cli':{'intermediate_mod':['_DAPI.png','_Hema.png','_Lap2.png','_Marker.png'],
                                         'intermediate_seg':['_DAPI_s.png'],
                                         'seg':['_Seg.png'],'postprocessed':['_SegOverlaid.png','_SegRefined.png']}}
                        },
              'ext':{'model':'DeepLIIFExt',
                     'dir_input_train':['Datasets/Sample_Dataset_ext_withseg','Datasets/Sample_Dataset_ext_noseg'],
                     'dir_input_testpy':['Datasets/Sample_Dataset_ext_withseg','Datasets/Sample_Dataset_ext_noseg'],
                     'dir_input_inference':['Datasets/Sample_Dataset_ext_withseg/test_cli','Datasets/Sample_Dataset_ext_noseg/test_cli'],
                     'dir_model':['../checkpoints/deepliif_extension_LN_Tonsil_4mod_400epochs','../checkpoints/HER2_5mod_400epochs'],
                     'modalities_no':[4,5],
                     'seg_gen':[True,False],
                     'tile_size':1024,
                     'baseline':['c1e7aee'],
                     'dir_output_baseline_testpy':['Datasets_baseline/Sample_Dataset_ext_withseg/test','Datasets_baseline/Sample_Dataset_ext_noseg/test'],
                     'dir_output_baseline_inference':['Datasets_baseline/Sample_Dataset_ext_withseg/test_cli','Datasets_baseline/Sample_Dataset_ext_noseg/test_cli'],
                     'suffix':{'testpy':{'intermediate_mod':['_B_1.png','_B_2.png','_B_3.png','_B_4.png','_B_5.png'],
                                         'seg':['_BS_1.png','_BS_2.png','_BS_3.png','_BS_4.png','_BS_5.png']},
                                  'cli':{'intermediate_mod':['_mod1.png','_mod2.png','_mod3.png','_mod4.png','_mod5.png'],
                                         'seg':['_Seg1.png','_Seg2.png','_Seg3.png','_Seg4.png','_Seg5.png'],
                                         'postprocessed':['_Overlaid.png','_Refined.png']}}
                     },
              'sdg':{'model':'SDG',
                     'dir_input_train':['Datasets/Sample_Dataset_sdg'],
                     'dir_input_testpy':['Datasets/Sample_Dataset_sdg'],
                     'dir_input_inference':['Datasets/Sample_Dataset_sdg/test_cli'],
                     'dir_model':['../checkpoints/sdg_20240104'],
                     'modalities_no': [4],
                     'seg_gen':[False],
                     'tile_size':512,
                     'baseline':['tests-image-quality-baseline-sdg'],
                     'dir_output_baseline_testpy':['Datasets_baseline/Sample_Dataset_sdg/test'],
                     'dir_output_baseline_inference':['Datasets_baseline/Sample_Dataset_sdg/test_cli'],
                     'suffix':{'testpy':{'intermediate_mod':['_B_1.png','_B_2.png','_B_3.png','_B_4.png','_B_5.png']},
                                  'cli':{'intermediate_mod':['_mod1.png','_mod2.png','_mod3.png','_mod4.png','_mod5.png']}}
                     },
              'cyclegan':{'model':'CycleGAN',
                     'dir_input_train':['Datasets/Sample_Dataset_cyclegan'],
                     'dir_input_testpy':['Datasets/Sample_Dataset_cyclegan'],
                     'dir_input_inference':['Datasets/Sample_Dataset_cyclegan/test_cli'],
                     'dir_model':['../checkpoints/pasto3mods_test'],
                     'modalities_no': [1],
                     'seg_gen':[False],
                     'tile_size':512,
                     'baseline':['tests-image-quality-baseline-cyclegan'],
                     'dir_output_baseline_testpy':['Datasets_baseline/Sample_Dataset_cyclegan/test'],
                     'dir_output_baseline_inference':['Datasets_baseline/Sample_Dataset_cyclegan/test_cli'],
                     'suffix':{'testpy':{'intermediate_mod':['_Bs_1.png']},
                                  'cli':{'intermediate_mod':['_GA_1.png']}}
                     }}

def pytest_addoption(parser):
    parser.addoption("--model_type", action="store", default="latest")

import pytest

@pytest.fixture(scope="session")
def model_type(pytestconfig):
    return pytestconfig.getoption("model_type")

@pytest.fixture(scope="session")
def model_info(model_type):
    return MODEL_INFO[model_type]

@pytest.fixture(scope="session")
def model_dir(model_info):
    return model_info['dir_model']

@pytest.fixture(scope="session")
def foldername_suffix(model_info):
    if model_info['model'] in ['CycleGAN']:
        return 'A' # trainA, testA, ...
    else:
        return '' # train, test, ...