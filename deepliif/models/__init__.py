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
from deepliif.util.util import tensor_to_pil, check_multi_scale
from deepliif.data import transform
from deepliif.postprocessing import adjust_marker, adjust_dapi, compute_IHC_scoring, \
    overlay_final_segmentation_mask, create_final_segmentation_mask_with_boundaries, create_basic_segmentation_mask

from .base_model import BaseModel

# import for init purpose, not used in this script
from .DeepLIIF_model import DeepLIIFModel



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
    net = torch.jit.load(model_pt_path, map_location=device)
    net = disable_batchnorm_tracking_stats(net)
    net.eval()
    return net



def load_eager_models(model_dir, devices, opt):
    # create a model given model and other options
    model = create_model(opt)
    # regular setup: load and print networks; create schedulers
    model.setup(opt)

    nets = {}
    for n in ['G1', 'G2', 'G3', 'G4','G51', 'G52', 'G53', 'G54', 'G55']:
        net = getattr(model,'net'+n)
        net.load_state_dict(torch.load(
            os.path.join(model_dir, f'latest_net_{n}.pth'),
            map_location=devices[n]
        ))
        nets[n] = disable_batchnorm_tracking_stats(net)
        nets[n].eval()

    return nets


@lru_cache
def init_nets(model_dir, eager_mode=False, opt=None):
    """
    Init DeepLIIF networks so that every net in
    the same group is deployed on the same GPU
    """
    net_groups = [
        ('G1', 'G52'),
        ('G2', 'G53'),
        ('G3', 'G54'),
        ('G4', 'G55'),
        ('G51',)
    ]

    number_of_gpus = torch.cuda.device_count()
    # number_of_gpus = 0
    if number_of_gpus:
        chunks = [itertools.chain.from_iterable(c) for c in chunker(net_groups, number_of_gpus)]
        # chunks = chunks[1:]
        devices = {n: torch.device(f'cuda:{i}') for i, g in enumerate(chunks) for n in g}
    else:
        devices = {n: torch.device('cpu') for n in itertools.chain.from_iterable(net_groups)}

    if eager_mode:
        return load_eager_models(model_dir, devices, opt)

    return {
        n: load_torchscript_model(os.path.join(model_dir, f'{n}.pt'), device=d)
        for n, d in devices.items()
    }


def compute_overlap(img_size, tile_size):
    w, h = img_size
    if round(w / tile_size) == 1 and round(h / tile_size) == 1:
        return 0

    return tile_size // 4


def run_torchserve(img, model_path=None, eager_mode=False, opt=None):
    """
    eager_mode: not used in this function; put in place to be consistent with run_dask
           so that run_wrapper() could call either this function or run_dask with
           same syntax
    opt: same as eager_mode
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


def run_dask(img, model_path, eager_mode=False, opt=None):
    model_dir = os.getenv('DEEPLIIF_MODEL_DIR', model_path)
    nets = init_nets(model_dir, eager_mode, opt)

    ts = transform(img.resize((512, 512)))

    @delayed
    def forward(input, model):
        with torch.no_grad():
            return model(input.to(next(model.parameters()).device))

    seg_map = {'G1': 'G52', 'G2': 'G53', 'G3': 'G54', 'G4': 'G55'}

    lazy_gens = {k: forward(ts, nets[k]) for k in seg_map}
    gens = compute(lazy_gens)[0]

    lazy_segs = {v: forward(gens[k], nets[v]).to(torch.device('cpu')) for k, v in seg_map.items()}
    lazy_segs['G51'] = forward(ts, nets['G51']).to(torch.device('cpu'))
    segs = compute(lazy_segs)[0]

    seg_weights = [0.25, 0.25, 0.25, 0, 0.25]
    seg = torch.stack([torch.mul(n, w) for n, w in zip(segs.values(), seg_weights)]).sum(dim=0)

    res = {k: tensor_to_pil(v) for k, v in gens.items()}
    res['G5'] = tensor_to_pil(seg)

    return res


def is_empty(tile):
    # return True if np.mean(np.array(tile) - np.array(mean_background_val)) < 40 else False
    return True if calculate_background_area(tile) > 98 else False


def run_wrapper(tile, run_fn, model_path, eager_mode=False, opt=None):
    if is_empty(tile):
        return {
            'G1': Image.new(mode='RGB', size=(512, 512), color=(201, 211, 208)),
            'G2': Image.new(mode='RGB', size=(512, 512), color=(10, 10, 10)),
            'G3': Image.new(mode='RGB', size=(512, 512), color=(0, 0, 0)),
            'G4': Image.new(mode='RGB', size=(512, 512), color=(10, 10, 10)),
            'G5': Image.new(mode='RGB', size=(512, 512), color=(0, 0, 0))
        }
    else:
        return run_fn(tile, model_path, eager_mode, opt)


def inference_old(img, tile_size, overlap_size, model_path, use_torchserve=False, eager_mode=False,
                  color_dapi=False, color_marker=False):
    
    tiles = list(generate_tiles(img, tile_size, overlap_size))

    run_fn = run_torchserve if use_torchserve else run_dask
    # res = [Tile(t.i, t.j, run_fn(t.img, model_path)) for t in tiles]
    res = [Tile(t.i, t.j, run_wrapper(t.img, run_fn, model_path, eager_mode)) for t in tiles]

    def get_net_tiles(n):
        return [Tile(t.i, t.j, t.img[n]) for t in res]

    images = {}

    images['Hema'] = stitch(get_net_tiles('G1'), tile_size, overlap_size).resize(img.size)

    # images['DAPI'] = stitch(
    #     [Tile(t.i, t.j, adjust_background_tile(dt.img))
    #      for t, dt in zip(tiles, get_net_tiles('G2'))],
    #     tile_size, overlap_size).resize(img.size)
    # dapi_pix = np.array(images['DAPI'])
    # dapi_pix[:, :, 0] = 0
    # images['DAPI'] = Image.fromarray(dapi_pix)

    images['DAPI'] = stitch(get_net_tiles('G2'), tile_size, overlap_size).resize(img.size)
    dapi_pix = np.array(images['DAPI'].convert('L').convert('RGB'))
    if color_dapi:
      dapi_pix[:, :, 0] = 0
    images['DAPI'] = Image.fromarray(dapi_pix)
    images['Lap2'] = stitch(get_net_tiles('G3'), tile_size, overlap_size).resize(img.size)
    images['Marker'] = stitch(get_net_tiles('G4'), tile_size, overlap_size).resize(img.size)
    marker_pix = np.array(images['Marker'].convert('L').convert('RGB'))
    if color_marker:
      marker_pix[:, :, 2] = 0
    images['Marker'] = Image.fromarray(marker_pix)

    # images['Marker'] = stitch(
    #     [Tile(t.i, t.j, kt.img)
    #      for t, kt in zip(tiles, get_net_tiles('G4'))],
    #     tile_size, overlap_size).resize(img.size)

    images['Seg'] = stitch(get_net_tiles('G5'), tile_size, overlap_size).resize(img.size)

    return images


def inference(img, tile_size, overlap_size, model_path, use_torchserve=False, eager_mode=False,
              color_dapi=False, color_marker=False, opt=None):

    rescaled, rows, cols = format_image_for_tiling(img, tile_size, overlap_size)

    run_fn = run_torchserve if use_torchserve else run_dask

    images = {}
    images['Hema'] = create_image_for_stitching(tile_size, rows, cols)
    images['DAPI'] = create_image_for_stitching(tile_size, rows, cols)
    images['Lap2'] = create_image_for_stitching(tile_size, rows, cols)
    images['Marker'] = create_image_for_stitching(tile_size, rows, cols)
    images['Seg'] = create_image_for_stitching(tile_size, rows, cols)

    for i in range(cols):
        for j in range(rows):
            tile = extract_tile(rescaled, tile_size, overlap_size, i, j)
            res = run_fn(tile, model_path, eager_mode, opt)

            stitch_tile(images['Hema'], res['G1'], tile_size, overlap_size, i, j)
            stitch_tile(images['DAPI'], res['G2'], tile_size, overlap_size, i, j)
            stitch_tile(images['Lap2'], res['G3'], tile_size, overlap_size, i, j)
            stitch_tile(images['Marker'], res['G4'], tile_size, overlap_size, i, j)
            stitch_tile(images['Seg'], res['G5'], tile_size, overlap_size, i, j)

    images['Hema'] = images['Hema'].resize(img.size)
    images['DAPI'] = images['DAPI'].resize(img.size)
    images['Lap2'] = images['Lap2'].resize(img.size)
    images['Marker'] = images['Marker'].resize(img.size)
    images['Seg'] = images['Seg'].resize(img.size)

    if color_dapi:
        matrix = (       0,        0,        0, 0,
                  299/1000, 587/1000, 114/1000, 0,
                  299/1000, 587/1000, 114/1000, 0)
        images['DAPI'] = images['DAPI'].convert('RGB', matrix)

    if color_marker:
        matrix = (299/1000, 587/1000, 114/1000, 0,
                  299/1000, 587/1000, 114/1000, 0,
                         0,        0,        0, 0)
        images['Marker'] = images['Marker'].convert('RGB', matrix)

    return images


def postprocess(img, seg_img, thresh=80, noise_objects_size=20, small_object_size=50):
    mask_image = create_basic_segmentation_mask(np.array(img), np.array(seg_img),
                                                thresh, noise_objects_size, small_object_size)
    images = {}
    images['SegOverlaid'] = Image.fromarray(overlay_final_segmentation_mask(np.array(img), mask_image))
    images['SegRefined'] = Image.fromarray(create_final_segmentation_mask_with_boundaries(np.array(mask_image)))

    all_cells_no, positive_cells_no, negative_cells_no, IHC_score = compute_IHC_scoring(mask_image)
    scoring = {
        'num_total': all_cells_no,
        'num_pos': positive_cells_no,
        'num_neg': negative_cells_no,
        'percent_pos': IHC_score
    }

    return images, scoring


def infer_modalities(img, tile_size, model_dir, eager_mode=False,
                     color_dapi=False, color_marker=False, opt=None):
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
        model_path=model_dir,
        eager_mode=eager_mode,
        color_dapi=color_dapi,
        color_marker=color_marker,
        opt=opt
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
