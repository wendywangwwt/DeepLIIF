import os
import subprocess
import shutil
import time
from .ComputeStatistics import Statistics

from ..options.test_options import TestOptions
from ..options import read_model_params, Options, print_options
from ..data import create_dataset
from ..models import create_model, init_nets, infer_modalities, infer_results_for_wsi
from ..models import postprocess as _postprocess
from ..util.visualizer import save_images
from ..util import html, allowed_file
import torch
import click

from PIL import Image
import json
from ..models import postprocess
import re
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import ToPILImage

def ensure_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)
        


def get_cell_count_metrics(dir_img, dir_save=None, model = None,
                           tile_size=512, single_tile=False,
                           use_marker = False,
                           save_individual = False):
    """
    Obtain cell count metrics through postprocess functions.
    Currenlty implemented only for ground truth tiles.
    
    dir_img: directory to load images for calculation
    dir_save: directory to save the results out
    model: model type (DeepLIIF, DeepLIIFExt, SDG)
    tile_size: tile size used for postprocess calculation
    single_tile: True if the images are single-tile images; use False if the 
                 images contain a row of multiple tiles like those used in
                 training or validation
    use_marker: whether to use the marker image (if Truem, assumes the marker
                image is the second last tile (single_tile=False) or has a 
                suffix of "_4.png" (single_tile=True))
    """
    dir_save = dir_save if dir_img is None else dir_save
    
    if model is None:
        if 'sdg' in dir_img:
            model = 'SDG'
        elif 'ext' in dir_img:
            model = 'DeepLIIFExt'
        else:
            model = 'DeepLIIF'
    
    if single_tile:
        fns = [x for x in os.listdir(dir_img) if x.endswith('_5.png') or x.endswith('_4.png')]
        fns = list(set([x[:-6] for x in fns])) # fns do not have extention
    else:
        fns = [x for x in os.listdir(dir_img) if x.endswith('.png')] # fns have extension
    
    d_metrics = {}
    count = 0
    for fn in fns:
        if single_tile:
            img_gt = Image.open(os.path.join(dir_img,fn+'_5.png'))
            img_marker = Image.open(os.path.join(dir_img,fn+'_4.png'))
            img_input = Image.open(os.path.join(dir_img.replace('/gt','/input'),fn+'.png'))
            k = fn
        else:
            img = Image.open(os.path.join(dir_img,fn))
            w, h = img.size
            
            # assume in the row of tiles, the first is the input and the last is the ground truth
            img_input = img.crop((0,0,h,h))
            img_gt = img.crop((w-h,0,w,h))
            img_marker = img.crop((w-h*2,0,w-h,h)) # the second last is marker, if marker is included
            k = re.sub('\..*?$','',fn) # remove extension
    
        images = {'Seg':img_gt}
        if use_marker:
            images['Marker'] = img_marker
        
        post_images, scoring = postprocess(img_input, images, tile_size, model)
        d_metrics[k] = scoring
        
        if save_individual:
            with open(os.path.join(
                dir_save,
                k+'.json'
            ), 'w') as f:
                json.dump(scoring, f, indent=2)
        
        count += 1
        
        if count % 100 == 0 or count == len(fns):
            print(count,'/',len(fns))

    with open(os.path.join(
                dir_save,
                'metrics.json'
            ), 'w') as f:
                json.dump(d_metrics, f, indent=2)


def format_file_structure(output_dir,source_folder='images',target_folder={'gt':'_real_B', 'pred':'_fake_B', 'input':'_real_A'}):
    """
    target_folder: keys are new folder names, values are unique token with which to filter the needed images for this folder
        1) if segmentation metrics are needed and no seg images are generated from the main model, folder name "input" can be
        specified when you want to run segmentation (using the seg model) on the original input as well
        2) if you want to also run metrics on the original input (e.g., SSIM/PSNR for upscaling task), then you should also specify
        "input""
        otherwise only "gt" and "pred" are used
    """
    dir_source = os.path.join(output_dir,source_folder)
    
    for folder_name, unique_token in target_folder.items():
        # create gt_dir and pred_dir
        dir_folder = os.path.join(output_dir, folder_name)
        if os.path.exists(dir_folder) and os.path.isdir(dir_folder):
            shutil.rmtree(dir_folder)
        os.makedirs(dir_folder)

        # separate the images
        # avoid exit code 126 caused by too many files to copy
        subprocess.run(f'for i in {dir_source}/*{unique_token}*; do cp "$i" {dir_folder}; done', shell=True, check=True)
        # subprocess.run(f"cp {dir_source}/*{unique_token}* {dir_folder}", shell=True, check=True)
        
        # rename files
        fns = os.listdir(dir_folder)
        for fn in fns:
            os.rename(f'{dir_folder}/{fn}',f"{dir_folder}/{fn.replace(f'{unique_token}','_')}")


def generate_predictions(dataroot, results_dir, checkpoints_dir, name='', num_test=10000,
                         phase='val', gpu_ids=(-1,), batch_size=None, epoch='latest', postprocess=False, tile_size=512):
    """
    a function version of cli.py test / test.py, to be used with evaluate()
    params:
      dataroot: reads images from here; expected to have a subfolder
      results_dir: saves results here.
      checkpoints_dir: load models from here.
      name: name of the experiment, used as a subfolder under results_dir
      num_test: only run test for num_test images
      phase: this effectively refers to the subfolder name from where to load the images
      gpu_ids: gpu-ids 0 gpu-ids 1 or gpu-ids -1 for CPU
      batch_size: input batch size
      preprocess: call the preprocess function to generate SegRefined, SegOverlaid, and cell count json
      tile_size: tile size used to process the provided data
    """
    # retrieve options used in training setting, similar to cli.py test
    model_dir = os.path.join(checkpoints_dir, name)
    opt = Options(path_file=os.path.join(model_dir,'train_opt.txt'), mode='test')

    if gpu_ids and gpu_ids[0] == -1:
        gpu_ids = []

    number_of_gpus_all = torch.cuda.device_count()
    #print(number_of_gpus_all, len(gpu_ids))
    if number_of_gpus_all < len(gpu_ids) and -1 not in gpu_ids:
        print(f'Specified to use GPU {opt.gpu_ids} for inference, but there are only {number_of_gpus_all} GPU devices. Switched to CPU inference.')
        number_of_gpus = 0
        gpu_ids = [-1]

    # overwrite/supply unseen options using the values from the options provided in the command
    setattr(opt,'checkpoints_dir',checkpoints_dir)
    setattr(opt,'dataroot',dataroot)
    setattr(opt,'name',name)
    setattr(opt,'results_dir',results_dir)
    setattr(opt,'num_test',num_test)
    setattr(opt,'phase',phase)
    setattr(opt,'gpu_ids',gpu_ids)
    setattr(opt,'batch_size',batch_size)
    setattr(opt,'epoch',str(epoch))

    if not hasattr(opt,'seg_gen'): # old settings for DeepLIIF models
        opt.seg_gen = True

    # hard-code some parameters for test.py
    opt.aspect_ratio = 1.0 # from previous default setting
    opt.display_winsize = 512 # from previous default setting
    opt.use_dp = True # whether to initialize model in DataParallel setting (all models to one gpu, then pytorch controls the usage of specified set of GPUs for inference)
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    print_options(opt)

    batch_size = batch_size if batch_size else opt.batch_size
    dataset = create_dataset(opt, phase=phase, batch_size=batch_size)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    torch.backends.cudnn.benchmark = False
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    model.eval()
    # if opt.eval:
    #     model.eval()

    _start_time = time.time()
    print('generate_predictions postprocess:',postprocess)

    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break

        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s (batch size %s)' % (i*batch_size, img_path[-1], batch_size))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        
        if postprocess:
            # import pickle
            # pickle.dump(visuals,open('/userfs/DeepLIIF/visuals.pk','wb'))
            # post_images, scoring = postprocess(img, images, tile_size, opt.model)
            # first dimension is batch size; .permute(1, 2, 0) changes tensor shape from (3,512,512) to (512,512,3)
            # visuals & data tensor value range: -1 to 1, needs to rescale to 0 to 1 for ToPILImage() to function properly
            l_seg_names = ['fake_B_5']
            assert l_seg_names[0] in visuals.keys(), f'Cannot find {l_seg_names[0]} in generated image names ({list(visuals.keys())})'
            seg_mod_suffix = l_seg_names[0].split('_')[-1]
            l_seg_names += [x for x in visuals.keys() if x.startswith('fake') and x.split('_')[-1].startswith(seg_mod_suffix) and x != l_seg_names[0]]
            # print(f'Running postprocess for {len(l_seg_names)} generated images ({l_seg_names})')
            
            for seg_name in l_seg_names:
                images = {'Seg':ToPILImage()((visuals[seg_name][0].cpu()+1)/2),
                          'Marker':ToPILImage()((visuals['fake_B_4'][0].cpu()+1)/2)} 
                post_images, scoring = _postprocess(ToPILImage()((data['A'][0]+1)/2), images, tile_size, opt.model)
                images = {**images, **post_images}
                #print(scoring)
                
                filename = img_path[0].split('/')[-1]
                # print()
                for name, i in images.items():
                    i.save(os.path.join(
                        web_dir,'images',
                        filename.replace('.' + filename.split('.')[-1], f'_{seg_name}_{name}.png')
                    ))
    
                with open(os.path.join(
                        web_dir,'images',
                        filename.replace('.' + filename.split('.')[-1], f'_{seg_name}.json')
                ), 'w') as f:
                    json.dump(scoring, f, indent=2)

    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour, t_min, t_sec))
    webpage.save()  # save the HTML
    
    


# def generate_predictions(dataroot, results_dir, checkpoints_dir, name='', num_test=10000,
#                          phase='val', gpu_ids=(-1,), batch_size=None, epoch='latest'):
#     """
#     a function version of cli.py test / test.py, to be used with evaluate()
#     params:
#       dataroot: reads images from here; expected to have a subfolder
#       results_dir: saves results here.
#       checkpoints_dir: load models from here.
#       name: name of the experiment, used as a subfolder under results_dir
#       num_test: only run test for num_test images
#       phase: this effectively refers to the subfolder name from where to load the images
#       gpu_ids: gpu-ids 0 gpu-ids 1 or gpu-ids -1 for CPU
#       batch_size: input batch size
#     """
#     input_dir = os.path.join(dataroot,phase) 
#     output_dir = results_dir
#     tile_size = None
#     model_dir = checkpoints_dir
#     region_size=20000
#     eager_mode=True
#     color_dapi=False
#     color_marker=False
#     save_type_pattern = ''#'Seg,Hema'
#     
#     l_save_type_pattern = save_type_pattern.split(',')
#                                    
#     output_dir = output_dir or input_dir
#     if os.path.exists(output_dir):
#         print(f'Folder {output_dir} already exists')
#         # print(f'overwrite: {overwrite}')
#         # if not overwrite:
#         #     print('Skip prediction generation')
#         #     return
# 
#     ensure_exists(output_dir)
# 
#     image_files = sorted([fn for fn in os.listdir(input_dir) if allowed_file(fn)])
#     print(len(image_files),'images found')
#     files = os.listdir(model_dir)
#     assert 'train_opt.txt' in files, f'file train_opt.txt is missing from model directory {model_dir}'
#     opt = Options(path_file=os.path.join(model_dir,'train_opt.txt'), mode='test')
#     opt.use_dp = False
#     
#     number_of_gpus_all = torch.cuda.device_count()
#     print(number_of_gpus_all, len(gpu_ids))
#     if number_of_gpus_all < len(gpu_ids) and -1 not in gpu_ids:
#         print(f'Specified to use GPU {opt.gpu_ids} for inference, but there are only {number_of_gpus_all} GPU devices. Switched to CPU inference.')
#         number_of_gpus = 0
#         gpu_ids = [-1]
#         
#     if len(gpu_ids) > 0 and gpu_ids[0] == -1:
#         gpu_ids = []
#     elif len(gpu_ids) == 0:
#         gpu_ids = list(range(number_of_gpus_all))
#     
#     opt.gpu_ids = gpu_ids # overwrite gpu_ids; for test command, default gpu_ids at first is [] which will be translated to a list of all gpus
#     
#     # fix opt from old settings
#     if not hasattr(opt,'modalities_no') and hasattr(opt,'targets_no'):
#         opt.modalities_no = opt.targets_no - 1
#         del opt.targets_no
#     print_options(opt)
# 
#     count = 0
#     with click.progressbar(
#             image_files,
#             label=f'Processing {len(image_files)} images',
#             item_show_func=lambda fn: fn
#     ) as bar:
#         for filename in bar:
#             if '.svs' in filename:
#                 start_time = time.time()
#                 infer_results_for_wsi(input_dir, filename, output_dir, model_dir, tile_size, region_size)
#                 print(time.time() - start_time)
#             else:
#                 img = Image.open(os.path.join(input_dir, filename)).convert('RGB')
#                 images, scoring = infer_modalities(img, tile_size, model_dir, eager_mode, color_dapi, color_marker, opt)
# 
#                 for name, i in images.items():
#                     if len(l_save_type_pattern) == 0 or (len(l_save_type_pattern) > 0 and any(s in name for s in l_save_type_pattern)): # if nothing is specified, or if any specified elements matches the name
#                         i.save(os.path.join(
#                             output_dir,
#                             filename.replace('.' + filename.split('.')[-1], f'_{name}.png')
#                         ))
# 
#                 with open(os.path.join(
#                         output_dir,
#                         filename.replace('.' + filename.split('.')[-1], f'.json')
#                 ), 'w') as f:
#                     json.dump(scoring, f, indent=2)
#                 
#             count += 1
#             if count % 100 == 0 or count == len(image_files):
#                 print(f'Done {count}/{len(image_files)}')
# 

    

def generate_predictions_inference(input_dir, output_dir, tile_size=None, model_dir='./model-server/DeepLIIF_Latest_Model/', 
                                   gpu_ids=(-1,), region_size=20000, eager_mode=True, color_dapi=False, color_marker=False,
                                   save_type_pattern='Seg',overwrite=False):
    output_dir = output_dir or input_dir
    if os.path.exists(output_dir):
        print(f'Folder {output_dir} already exists')
        print(f'overwrite: {overwrite}')
        if not overwrite:
            print('Skip prediction generation')
            return

    ensure_exists(output_dir)

    image_files = sorted([fn for fn in os.listdir(input_dir) if allowed_file(fn)])
    print(len(image_files),'images found')
    files = os.listdir(model_dir)
    assert 'train_opt.txt' in files, f'file train_opt.txt is missing from model directory {model_dir}'
    opt = Options(path_file=os.path.join(model_dir,'train_opt.txt'), mode='test')
    opt.use_dp = False
    
    number_of_gpus_all = torch.cuda.device_count()
    print(number_of_gpus_all, len(gpu_ids))
    if number_of_gpus_all < len(gpu_ids) and -1 not in gpu_ids:
        print(f'Specified to use GPU {opt.gpu_ids} for inference, but there are only {number_of_gpus_all} GPU devices. Switched to CPU inference.')
        number_of_gpus = 0
        gpu_ids = [-1]
        
    if len(gpu_ids) > 0 and gpu_ids[0] == -1:
        gpu_ids = []
    elif len(gpu_ids) == 0:
        gpu_ids = list(range(number_of_gpus_all))
    
    opt.gpu_ids = gpu_ids # overwrite gpu_ids; for test command, default gpu_ids at first is [] which will be translated to a list of all gpus
    
    # fix opt from old settings
    if not hasattr(opt,'modalities_no') and hasattr(opt,'targets_no'):
        opt.modalities_no = opt.targets_no - 1
        del opt.targets_no
    print_options(opt)

    count = 0
    with click.progressbar(
            image_files,
            label=f'Processing {len(image_files)} images',
            item_show_func=lambda fn: fn
    ) as bar:
        for filename in bar:
            if '.svs' in filename:
                start_time = time.time()
                infer_results_for_wsi(input_dir, filename, output_dir, model_dir, tile_size, region_size)
                print(time.time() - start_time)
            else:
                img = Image.open(os.path.join(input_dir, filename)).convert('RGB')
                images, scoring = infer_modalities(img, tile_size, model_dir, eager_mode, color_dapi, color_marker, opt)

                for name, i in images.items():
                    if save_type_pattern and save_type_pattern in name:
                        i.save(os.path.join(
                            output_dir,
                            filename.replace('.' + filename.split('.')[-1], f'_{name}.png')
                        ))

                with open(os.path.join(
                        output_dir,
                        filename.replace('.' + filename.split('.')[-1], f'.json')
                ), 'w') as f:
                    json.dump(scoring, f, indent=2)
                
            count += 1
            if count % 100 == 0 or count == len(image_files):
                print(f'Done {count}/{len(image_files)}')
    

def evaluate(model_dir, input_dir, output_dir, seg_model_dir=None, phase='val', subfolder=None, image_types='B_1,B_2,B_3,B_4', seg_type='B_5', 
         target_folder={'gt':'_real_B', 'pred':'_fake_B', 'input':'_real_A'},
         seg_gen=False, epoch='latest',mode='Segmentation', seg_on_input=False, include_input_metrics=False, batch_size=None, gpu_ids=(-1,),
         overwrite=False, 
         postprocess=False,verbose=False):
    """
    params:
      model_dir: directory of the trained model
      input_dir: input directory of the whole dataset, which should contain multiple subfolders like train and test
      output_dir: path to the trained model
      seg_model_dir: only used when seg_gen=False; this is the model to generate segmentation mask for both gt and pred images
      phase: this effectively is the subfolder name under model_dir in which is the input data
      subfolder: this effectively is the subfolder name under output_dir in which is the folder for predictions "./images" locates
      image_types: unique marker in filename for each modality; in model class DeepLIIFExt and SDG, it could be a string like B_1,B_2,B_3
      seg_type: unique marker in filename for segmentation, if exists; in model class DeepLIIFExt and SDG, it could be a string like B_4
      seg_gen: True (Translation and Segmentation), False (Only Translation).
      epoch: the name of epoch to load, default to "latest"
      mode: Mode of the statistics computation including Segmentation, ImageSynthesis, All, SSIM, Upscaling
      seg_on_input: a flag to indicate whether to run the seg model on the input image; this applies only to the situation where 
            seg_gen=False, mode includes segmentation metrics, and seg_model_dir is provided; if True, 2 sets of segmentation metrics will
            be returned (seg on fake vs seg on real, seg on input vs seg on real)
      batch_size: batch size for inference; if not set, default is opt.batch_size
      overwrite: overwrite results; otherwise, if output_dir already exists, prediction generation will be skipped
      verbose: print more info if True
      target_folder: used to reformat folder structure
    """
    if seg_gen == False and mode in ['Segmentation','All']:
        if not seg_model_dir:
            mode = 'SSIM'
            print('seg_gen is False, cannot run segmentation metrics; mode is changed to SSIM')
    
    d_res = {'elapsed_time':[]}
    if verbose:
        params = locals()
        for k,v in params:
            print(k,v)
    
    if not subfolder:
        subfolder = f'{phase}_{epoch}'
    
    print('Generating predictions...')
    time_s = time.time()
    if os.path.exists(os.path.join(output_dir, subfolder)):
        print(f'Folder {os.path.join(output_dir, subfolder)} already exists')
        print(f'overwrite: {overwrite}')
        if not overwrite:
            print('Skip prediction generation')
        else:
            print('Deleting folder and regenerating predictions...')
            shutil.rmtree(output_dir)
            generate_predictions(checkpoints_dir=model_dir, name='.', dataroot=input_dir,
                         results_dir=output_dir, gpu_ids=gpu_ids, phase=phase, 
                         batch_size=batch_size, epoch=epoch, postprocess=postprocess)
    else:
        generate_predictions(checkpoints_dir=model_dir, name='.', dataroot=input_dir,
                         results_dir=output_dir, gpu_ids=gpu_ids, phase=phase,
                         batch_size=batch_size, epoch=epoch, postprocess=postprocess)
    d_res['elapsed_time'].append(time.time() - time_s)
    
    # if seg_gen == False and seg_model_dir:
    #     print('Generating segmentation mask...')
    #     time_s = time.time()
    #     seg_output_dir = os.path.join(output_dir, subfolder, 'seg')
    #     if os.path.exists(seg_output_dir):
    #         print(f'Folder {os.path.join(output_dir, subfolder, "seg")} already exists')
    #         print(f'overwrite: {overwrite}')
    #         if not overwrite:
    #             print('Skip segmentation mask generation')
    #         else:
    #             print('Deleting folder and regenerating segmentation mask...')
    #             shutil.rmtree(os.path.join(output_dir, subfolder, 'seg'))
    #             generate_predictions_inference(input_dir=os.path.join(output_dir,subfolder,'images'),
    #                                            output_dir=seg_output_dir, tile_size=512, model_dir=seg_model_dir,
    #                                            gpu_ids=gpu_ids, eager_mode=True)
    #     else:
    #         generate_predictions_inference(input_dir=os.path.join(output_dir,subfolder,'images'),
    #                                            output_dir=seg_output_dir, tile_size=512, model_dir=seg_model_dir,
    #                                            gpu_ids=gpu_ids, eager_mode=True)
    #     d_res['elapsed_time'].append(time.time() - time_s)

    print('Preparing folder structure and formating filenames...')
    time_s = time.time()
    format_file_structure(os.path.join(output_dir,subfolder),target_folder=target_folder) # the folder name val_latest
    d_res['elapsed_time'].append(time.time() - time_s)

    if seg_gen == False and seg_model_dir:
        print('Preparing folder structure and formating filenames...')
        time_s = time.time()
        format_file_structure(os.path.join(output_dir,subfolder),source_folder='seg',target_folder={'gt_seg':'_real_B_1_SegRefined','pred_seg':'_fake_B_1_SegRefined', 'input_seg':'_real_A_SegRefined'}) # the folder name val_latest
        d_res['elapsed_time'].append(time.time() - time_s)


    print('Starting ComputeStatistics.py...')

    time_s = time.time()
    gt_dir = os.path.join(output_dir, subfolder, 'gt')
    pred_dir = os.path.join(output_dir, subfolder, 'pred')
    raw_dir = os.path.join(output_dir, subfolder, 'images')
    print(raw_dir)
    if seg_gen == False and seg_model_dir and mode in ['Segmentation', 'All']:
        gt_dir = os.path.join(output_dir, subfolder, 'gt_seg')
        pred_dir = os.path.join(output_dir, subfolder, 'pred_seg')
        
        stats = Statistics(gt_path=raw_dir, model_path=raw_dir, output_path=output_dir,
                           image_types=image_types, seg_type=seg_type, mode='Segmentation')
        d_stat = stats.run()

        if mode == 'All':
            gt_dir = os.path.join(output_dir, subfolder, 'gt')
            pred_dir = os.path.join(output_dir, subfolder, 'pred')
            stats = Statistics(gt_path=gt_dir, model_path=pred_dir, output_path=output_dir,
                               image_types=image_types, seg_type=seg_type, mode='ImageSynthesis')
            d_stat_imagesynthesis = stats.run()
            d_stat = {**d_stat, **d_stat_imagesynthesis}
    else:
        if mode == 'Segmentation':
            print('gt_path',raw_dir)
            print('model_path',raw_dir)
            print('image_types',image_types)
            print('seg_type',seg_type)
            stats = Statistics(gt_path=raw_dir, model_path=raw_dir, output_path=output_dir,
                           image_types=image_types, seg_type=seg_type, mode=mode)
            d_stat = stats.run()
        else: # currently this would fail for All mode because Segmentation uses different folder
            stats = Statistics(gt_path=gt_dir, model_path=pred_dir, output_path=output_dir,
                               image_types=image_types, seg_type=seg_type, mode=mode)
            d_stat = stats.run()

        if include_input_metrics:
            input_dir = os.path.join(output_dir, subfolder, 'input')
            stats = Statistics(gt_path=gt_dir, model_path=input_dir, output_path=output_dir,
                           image_types=image_types, seg_type=seg_type, mode=mode)
    d_res['elapsed_time'].append(time.time() - time_s)


    return {**d_stat, **d_res}

