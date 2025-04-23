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
from deepliif.options import read_model_params
from deepliif.data import create_dataset
from deepliif.models import create_model
from deepliif.util.visualizer import save_images
from deepliif.util import html


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    
    # retrieve options used in training setting, similar to 
    # https://github.com/nadeemlab/DeepLIIF/blob/cc4deffca64b8865415ba665290d7971b821b1bd/deepliif/models/__init__.py#L115
    # model_dir in init_nets() is equivalent to save_dir in basemodel init
    # https://github.com/nadeemlab/DeepLIIF/blob/cc4deffca64b8865415ba665290d7971b821b1bd/deepliif/models/base_model.py#L36
    model_dir = os.path.join(opt.checkpoints_dir, opt.name)
    files = os.listdir(model_dir)
    for f in files:
        if 'train_opt.txt' in f:
            param_dict = read_model_params(os.path.join(model_dir, f))
            opt.input_nc = int(param_dict['input_nc'])
            opt.output_nc = int(param_dict['output_nc'])
            opt.ngf = int(param_dict['ngf'])
            opt.norm = param_dict['norm']
            # opt.use_dropout = False if param_dict['no_dropout'] == 'True' else True # in DeepLIIFModel(), the option used is opt.no_dropout not opt.use_dropout
            # opt.padding_type = param_dict['padding'] # in DeepLIIFModel(), the option used is opt.padding not opt.padding_type
            opt.padding = param_dict['padding']
            
            if 'modalities_no' in param_dict:
                opt.modalities_no = int(param_dict['modalities_no'])
            if 'seg_gen' in param_dict:
                opt.seg_gen = bool(param_dict['seg_gen'].strip().lower()=='true')
            for attr_name in ['net_g','net_d','net_gs','net_ds']:
                if attr_name in param_dict:
                    setattr(opt,attr_name,param_dict[attr_name])
    
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
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
