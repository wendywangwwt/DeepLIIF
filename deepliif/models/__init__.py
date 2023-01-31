"""This package contains modules related to objective functions, optimizations, and network architectures.

To add a custom model class called 'dummy', you need to add a file called 'dummy_model.py' and define a subclass DummyModel inherited from BaseModel.
You need to implement the following five functions:
    -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
    -- <set_input>:                     unpack data from dataset and apply preprocessing.
    -- <forward>:                       produce intermediate results.
    -- <optimize_parameters>:           calculate loss, gradients, and update network weights.
    -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.

In the function <__init__>, you need to define four lists:
    -- self.loss_names (str list):          specify the training losses that you want to plot and save.
    -- self.model_names (str list):         define networks used in our training.
    -- self.visual_names (str list):        specify the images that you want to display and save.
    -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an usage.

Now you can use the model class by specifying flag '--model dummy'.
See our template model class 'template_model.py' for more details.
"""
import base64
import os
import itertools
import importlib
from functools import lru_cache
from io import BytesIO

import requests
import torch
from PIL import Image
import numpy as np
from dask import delayed, compute

from deepliif.util import *
from deepliif.util.util import tensor_to_pil
from deepliif.data import transform
from deepliif.postprocessing import adjust_marker, adjust_dapi, compute_IHC_scoring, \
    overlay_final_segmentation_mask, create_final_segmentation_mask_with_boundaries, create_basic_segmentation_mask

from .base_model import BaseModel
from .DeepLIIF_model import DeepLIIFModel
from .networks import get_norm_layer, ResnetGenerator, UnetGenerator


def find_model_using_name(model_name):
    """Import the module "models/[model_name]_model.py".

    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseModel,
    and it is case-insensitive.
    """
    model_filename = "deepliif.models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
                and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (
            model_filename, target_model_name))
        exit(0)

    return model


def get_option_setter(model_name):
    """Return the static method <modify_commandline_options> of the model class."""
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_model(opt):
    """Create a model given the option.

    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from deepliif.models import create_model
        >>> model = create_model(opt)
    """
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print("model [%s] was created" % type(instance).__name__)
    return instance


def load_torchscript_model(model_pt_path, device):
    return torch.jit.load(model_pt_path, map_location=device)


def read_model_params(file_addr):
    with open(file_addr) as f:
        lines = f.readlines()
    param_dict = {}
    for line in lines:
        if ':' in line:
            key = line.split(':')[0].strip()
            val = line.split(':')[1].split('[')[0].strip()
            if 'gpu_ids' in key:
                val = val.replace('(', '').replace(')', '')
            param_dict[key] = val.split(',') if ',' in val else val
    return param_dict


def read_train_options(model_dir):
    files = os.listdir(model_dir)
    param_dict = None
    for f in files:
        if 'train_opt.txt' in f:
            param_dict = read_model_params(os.path.join(model_dir, f))
    return param_dict


def load_eager_models(model_dir, devices):
    input_nc = 3
    output_nc = 3
    ngf = 64
    norm = 'batch'
    use_dropout = True
    padding_type = 'zero'
    modalities_no = 4
    seg_gen = True

    param_dict = read_train_options(model_dir)
    if param_dict:
        input_nc = int(param_dict['input_nc'])
        output_nc = int(param_dict['output_nc'])
        ngf = int(param_dict['ngf'])
        norm = param_dict['norm']
        use_dropout = False if param_dict['no_dropout'] == 'True' else True
        padding_type = param_dict['padding']
        modalities_no = int(param_dict['modalities_no'])
        seg_gen = (param_dict['seg_gen'] == 'True')
    
    param_dict['gpu_ids'] = 0
    norm_layer = get_norm_layer(norm_type=norm)
    nets = {}
    
    for i in range(1, modalities_no + 1):
        n = 'G_' + str(i)
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, padding_type=padding_type)
        net.eval()
        net.load_state_dict(torch.load(
            os.path.join(model_dir, f'latest_net_{n}.pth'),
            map_location=devices[n]
        ))
        nets[n] = disable_batchnorm_tracking_stats(net)

    if seg_gen:
        for i in range(1, modalities_no + 1):
            n = 'GS_' + str(i)
            net = UnetGenerator(input_nc * 3, output_nc, 9, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
            net.eval()
            net.load_state_dict(torch.load(
                os.path.join(model_dir, f'latest_net_{n}.pth'),
                map_location=devices[n]
            ))
            nets[n] = disable_batchnorm_tracking_stats(net)

    return nets


@lru_cache
def init_nets(model_dir, eager_mode=False):
    """
    Init DeepLIIF networks so that every net in
    the same group is deployed on the same GPU
    """
    param_dict = read_train_options(model_dir)
    modalities_no = int(param_dict['modalities_no']) if param_dict else 4
    seg_gen = (param_dict['seg_gen'] == 'True') if param_dict else True
    net_groups = []
    for i in range(1, modalities_no + 1):
        if seg_gen:
            net_groups.append(('G_' + str(i), 'GS_' + str(i)))
        else:
            net_groups.append(('G_' + str(i),))
    # net_groups.append(('GS_1',))

    number_of_gpus = torch.cuda.device_count()
    if number_of_gpus:
        chunks = [itertools.chain.from_iterable(c) for c in chunker(net_groups, number_of_gpus)]
        devices = {n: torch.device(f'cuda:{i}') for i, g in enumerate(chunks) for n in g}
    else:
        devices = {n: torch.device('cpu') for n in itertools.chain.from_iterable(net_groups)}

    if eager_mode:
        return load_eager_models(model_dir, devices)
    
    return {n:disable_batchnorm_tracking_stats(load_torchscript_model(os.path.join(model_dir, f'{n}.pt'), device=d)) for n, d in devices.items()}
    
    # # disable tracking running stats in batchnorm 
    # # https://discuss.pytorch.org/t/performance-highly-degraded-when-eval-is-activated-in-the-test-phase/3323/16
    # d_models = {}
    # for n, d in devices.items():
    #     model = load_torchscript_model(os.path.join(model_dir, f'{n}.pt'), device=d)
    #     for child in model.children():
    #         for ii in range(len(child)):
    #             if type(child[ii])==nn.BatchNorm2d:
    #                 child[ii].track_running_stats = False
    #     d_models[n] = model
    # return d_models


def compute_overlap(img_size, tile_size):
    w, h = img_size
    if round(w / tile_size) == 1 and round(h / tile_size) == 1:
        return 0

    return tile_size // 4


def run_torchserve(img, model_path=None, param_dict=None, eager_mode=False):
    """
    eager_mode: not used in this function; put in place to be consistent with run_dask
           so that run_wrapper() could call either this function or run_dask with
           same syntax
    """
    buffer = BytesIO()
    torch.save(transform(img.resize((512, 512))), buffer)

    torchserve_host = os.getenv('TORCHSERVE_HOST', 'http://localhost')
    res = requests.post(
        f'{torchserve_host}/wfpredict/deepliif',
        json={'img': base64.b64encode(buffer.getvalue()).decode('utf-8')}
    )

    res.raise_for_status()

    def deserialize_tensor(bs):
        return torch.load(BytesIO(base64.b64decode(bs.encode())), map_location=torch.device('cpu'))

    return {k: tensor_to_pil(deserialize_tensor(v)) for k, v in res.json().items()}


def run_dask(img, model_path, param_dict, eager_mode=False):
    model_dir = os.getenv('DEEPLIIF_MODEL_DIR', model_path)
    nets = init_nets(model_dir, eager_mode)

    modalities_no = int(param_dict['modalities_no']) if param_dict else 4
    seg_gen = (param_dict['seg_gen'] == 'True') if param_dict else True
    # seg_weights = list(map(float, param_dict['seg_weights'])) if param_dict else [1 / 3] * (modalities_no + 1)
    
    print('img.size() in run_dask',img.size)
    ts = transform(img.resize((1024, 1024)))
    # ts = transform(img)
    print('ts.size() in run_dask',ts.size())

    @delayed
    def forward(input, model):
        with torch.no_grad():
            return model(input.to(next(model.parameters()).device))

    # seg_map = {'G1': 'G52', 'G2': 'G53', 'G3': 'G54', 'G4': 'G55'}
    seg_map = {}
    for i in range(1, modalities_no + 1):
        seg_map['G_' + str(i)] = 'GS_' + str(i)

    lazy_gens = {k: forward(ts, nets[k]) for k in seg_map}
    gens = compute(lazy_gens)[0]
    res = {k: tensor_to_pil(v) for k, v in gens.items()}
    print('output from compute(lazy_gens) in run_dask',res['G_2'].size)

    if seg_gen:
        lazy_segs = {v: forward(torch.cat([ts.to(torch.device('cpu')), gens[next(iter(seg_map))].to(torch.device('cpu')), gens[k].to(torch.device('cpu'))], 1), nets[v]).to(torch.device('cpu')) for k, v in seg_map.items()}
        # lazy_segs['GS_1'] = forward(ts, nets['GS_1']).to(torch.device('cpu'))
        segs = compute(lazy_segs)[0]
        res.update({k: tensor_to_pil(v) for k, v in segs.items()})
        # res['GS_1'] = tensor_to_pil(segs.values()[0])
        # res['GS_2'] = tensor_to_pil(segs.values()[1])
        # for i in range(1, modalities_no + 1):
        #     seg = torch.stack([torch.mul(segs.values()[0], seg_weights[0]),
        #                             torch.mul(segs.values()[1], seg_weights[1]),
        #                             torch.mul(segs.values()[i], seg_weights[i])]).sum(dim=0)
        #     res['GS_' + str(i + 1)] = tensor_to_pil(seg)

    return res


def is_empty(tile):
    return True if np.mean(np.array(tile)) > 240 else False
    # return True if np.mean(np.array(tile) - np.array(mean_background_val)) < 40 else False
    # return True if calculate_background_area(tile) > 98 else False


def run_wrapper(tile, run_fn, model_path, param_dict, eager_mode=False):
    if is_empty(tile):
        res = {'G_' + str(i): Image.new(mode='RGB', size=(512, 512)) for i in range(1, int(param_dict['modalities_no']) + 1)}
        res.update({
            'GS_' + str(i): Image.new(mode='RGB', size=(512, 512)) for i in
            range(1, int(param_dict['modalities_no']) + 1)})
        return res
    else:
        return run_fn(tile, model_path, param_dict, eager_mode)


def inference(img, tile_size_center, tile_size, overlap_size, model_path, eager_mode=False, use_torchserve=False):
    param_dict = read_train_options(model_path)
    modalities_no = int(param_dict['modalities_no']) if param_dict else 4
    seg_gen = (param_dict['seg_gen'] == 'True') if param_dict else True
    overlap_size = int((tile_size - tile_size_center)/2)
    
    tiles = list(generate_tiles(img, tile_size_center, tile_size, overlap_size))
    
    import pickle
    with open('/userfs/tiles_v2.p','wb') as f:
        pickle.dump(tiles,f)
        
#     import sys
#     sys.exit(1)

    run_fn = run_torchserve if use_torchserve else run_dask
    res = [Tile(t.i, t.j, run_wrapper(t.img, run_fn, model_path, param_dict, eager_mode)) for t in tiles]

    def get_net_tiles(n):
        return [Tile(t.i, t.j, t.img[n]) for t in res]

    images = {}

    for i in range(1, modalities_no + 1):
        tiles_pred = get_net_tiles('G_' + str(i))
        images['mod' + str(i)] = stitch(tiles_pred, tile_size_center, tile_size, overlap_size).resize(img.size)
        with open('/userfs/tiles_pred_v2.p','wb') as f:
            pickle.dump(tiles_pred,f)

    if seg_gen:
        for i in range(1, modalities_no + 1):
            images['Seg' + str(i)] = stitch(get_net_tiles('GS_' + str(i)), tile_size_center, tile_size, overlap_size).resize(img.size)

    return images


def postprocess(img, images, thresh=80, noise_objects_size=20, small_object_size=50):
    processed_images = {}
    scoring = {}
    for img_name in list(images.keys()):
        if 'Seg' in img_name:
            seg_img = images[img_name]
            mask_image = create_basic_segmentation_mask(np.array(img), np.array(seg_img),
                                                        thresh, noise_objects_size, small_object_size)

            processed_images[img_name + '_Overlaid'] = Image.fromarray(overlay_final_segmentation_mask(np.array(img), mask_image))
            processed_images[img_name + '_Refined'] = Image.fromarray(create_final_segmentation_mask_with_boundaries(np.array(mask_image)))

            all_cells_no, positive_cells_no, negative_cells_no, IHC_score = compute_IHC_scoring(mask_image)
            scoring[img_name] = {
                'num_total': all_cells_no,
                'num_pos': positive_cells_no,
                'num_neg': negative_cells_no,
                'percent_pos': IHC_score
            }

    return processed_images, scoring


def infer_modalities(img, tile_size, model_dir):
    """
    This function is used to infer modalities for the given image using a trained model.
    :param img: The input image.
    :param tile_size: The tile size.
    :param model_dir: The directory containing serialized model files.
    :return: The inferred modalities and the segmentation mask.
    """
    if not tile_size:
        tile_size = check_multi_scale(Image.open('./images/target.png').convert('L'),
                                      img.convert('L'))
    tile_size = int(tile_size)

    images = inference(
        img,
        tile_size=tile_size,
        overlap_size=compute_overlap(img.size, tile_size),
        model_path=model_dir
    )

    post_images, scoring = postprocess(img, images['Seg'], small_object_size=20)
    images = {**images, **post_images}
    return images, scoring


def infer_results_for_wsi(input_dir, filename, output_dir, model_dir, tile_size, region_size=20000):
    """
    This function infers modalities and segmentation mask for the given WSI image. It

    :param input_dir: The directory containing the WSI.
    :param filename: The WSI name.
    :param output_dir: The directory for saving the inferred modalities.
    :param model_dir: The directory containing the serialized model files.
    :param tile_size: The tile size.
    :param region_size: The size of each individual region to be processed at once.
    :return:
    """
    results_dir = os.path.join(output_dir, filename)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    size_x, size_y, size_z, size_c, size_t, pixel_type = get_information(os.path.join(input_dir, filename))
    print(filename, size_x, size_y, size_z, size_c, size_t, pixel_type)
    results = {}
    start_x, start_y = 0, 0
    while start_x < size_x:
        while start_y < size_y:
            print(start_x, start_y)
            region_XYWH = (start_x, start_y, min(region_size, size_x - start_x), min(region_size, size_y - start_y))
            region = read_bioformats_image_with_reader(os.path.join(input_dir, filename), region=region_XYWH)

            region_modalities, region_scoring = infer_modalities(Image.fromarray((region * 255).astype(np.uint8)), tile_size, model_dir)

            for name, img in region_modalities.items():
                if name not in results:
                    results[name] = np.zeros((size_y, size_x, 3), dtype=np.uint8)
                results[name][region_XYWH[1]: region_XYWH[1] + region_XYWH[3],
                region_XYWH[0]: region_XYWH[0] + region_XYWH[2]] = np.array(img)
            start_y += region_size
        start_y = 0
        start_x += region_size

    write_results_to_pickle_file(os.path.join(results_dir, "results.pickle"), results)
    # read_results_from_pickle_file(os.path.join(results_dir, "results.pickle"))

    for name, img in results.items():
        write_big_tiff_file(os.path.join(results_dir, filename.replace('.svs', '_' + name + '.ome.tiff')), img,
                            tile_size)

    javabridge.kill_vm()
