"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import time
from deepliif.options.test_options import TestOptions
from deepliif.options import read_model_params, Options, print_options
from deepliif.data import create_dataset
from deepliif.models import create_model
from deepliif.util.visualizer import save_images
from deepliif.util import html
import torch
import click
    
@click.command()
#@click.option('--input-dir', default='./Sample_Large_Tissues/', help='reads images from here')
#@click.option('--output-dir', help='saves results here.')
#@click.option('--tile-size', default=None, help='tile size')
#@click.option('--model-dir', default='./model-server/DeepLIIF_Latest_Model/', help='load models from here.')
@click.option('--dataroot', required=True, help='reads images from here; expected to have a subfolder')
@click.option('--results_dir', required=True, help='saves results here.')
@click.option('--name', default='.', help='name of the experiment, used as a subfolder under results_dir')
@click.option('--checkpoints_dir', required=True, help='load models from here.')
@click.option('--num_test', default=10000, help='only run test for num_test images')
@click.option('--phase', default='test', help='this effectively refers to the subfolder name from where to load the images')
@click.option('--gpu_ids', type=int, multiple=True, help='gpu-ids 0 gpu-ids 1 or gpu-ids -1 for CPU')
@click.option('--batch_size', default=1, help='input batch size')
def test(dataroot, results_dir, name, checkpoints_dir, num_test, phase, gpu_ids, batch_size):
    # retrieve options used in training setting, similar to cli.py test
    model_dir = os.path.join(checkpoints_dir, name)
    opt = Options(path_file=os.path.join(model_dir,'train_opt.txt'), mode='test')
    
    if gpu_ids and gpu_ids[0] == -1:
        gpu_ids = []
    
    # overwrite/supply unseen options using the values from the options provided in the command
    setattr(opt,'checkpoints_dir',checkpoints_dir)
    setattr(opt,'dataroot',dataroot)
    setattr(opt,'name',name)
    setattr(opt,'results_dir',results_dir)
    setattr(opt,'num_test',num_test)
    setattr(opt,'phase',phase)
    setattr(opt,'gpu_ids',gpu_ids)
    setattr(opt,'batch_size',batch_size)
        
    if not hasattr(opt,'seg_gen'): # old settings for DeepLIIF models
        opt.seg_gen = True
    
    number_of_gpus_all = torch.cuda.device_count()
    if number_of_gpus_all < len(gpu_ids) and -1 not in gpu_ids:
        number_of_gpus = 0
        gpu_ids = [-1]
        print(f'Specified to use GPU {opt.gpu_ids} for inference, but there are only {number_of_gpus_all} GPU devices. Switched to CPU inference.')

    if len(gpu_ids) > 0 and gpu_ids[0] == -1:
        gpu_ids = []
    elif len(gpu_ids) == 0:
        gpu_ids = list(range(number_of_gpus_all))

    opt.gpu_ids = gpu_ids # overwrite gpu_ids; for test command, default gpu_ids at first is [] which will be translated to a list of all gpus
    
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
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
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

    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour, t_min, t_sec))
    webpage.save()  # save the HTML



if __name__ == '__main__':
    test()
    
