import subprocess
import os
import torch
import pytest

available_gpus = torch.cuda.device_count()

def test_cli_train(tmp_path, model_info):
    dirs_input = model_info['dir_input_train']
    for dir_input in dir_inputs:
        dir_save = tmp_path
        
        fns_input = [f for f in os.listdir(dir_input + '/train') if os.path.isfile(os.path.join(dir_input + '/train', f)) and f.endswith('png')]
        num_input = len(fns_input)
        assert num_input > 0
        
        res = subprocess.run(f'python cli.py train --model {model_info["model"]} --dataroot {dir_input} --name test_local --batch-size 1 --num-threads 0 --checkpoints-dir {dir_save} --remote True --n-epochs 1 --n-epochs-decay 1 --seg-gen {"noseg" not in dir_input}',shell=True)
        assert res.returncode == 0

        res = subprocess.run(f'python cli.py serialize --models-dir {dir_save} --output-dir {dir_save}',shell=True)
        assert res.returncode == 0
    

def test_cli_train_single_gpu(tmp_path, model_info, model_type):
    if torch.cuda.device_count() > 0:
        dir_inputs = model_info['dir_input_train']
        for dir_input in dir_inputs:
            dir_save = tmp_path
            
            fns_input = [f for f in os.listdir(dir_input + '/train') if os.path.isfile(os.path.join(dir_input + '/train', f)) and f.endswith('png')]
            num_input = len(fns_input)
            assert num_input > 0
            
            cmd = f'python cli.py train --model {model_info["model"]} --dataroot {dir_input} --name test_local --batch-size 1 --num-threads 0 --checkpoints-dir {dir_save} --remote True --n-epochs 1 --n-epochs-decay 1 --gpu-ids 0'
            if model_type == 'ext' and 'noseg' in dir_input:
                cmd += ' --seg-gen false'
            res = subprocess.run(cmd,shell=True)
            assert res.returncode == 0
    else:
        pytest.skip(f'Detected {available_gpus} (< 1) available GPUs. Skip.')
    

def test_cli_train_multi_gpu_dp(tmp_path, model_info, model_type):
    if torch.cuda.device_count() > 1:
        dir_inputs = model_info['dir_input_train']
        for dir_input in dir_inputs:
            dir_save = tmp_path
            
            fns_input = [f for f in os.listdir(dir_input + '/train') if os.path.isfile(os.path.join(dir_input + '/train', f)) and f.endswith('png')]
            num_input = len(fns_input)
            assert num_input > 0
            
            cmd = f'python cli.py train --model {model_info["model"]} --dataroot {dir_input} --name test_local --batch-size 1 --num-threads 0 --checkpoints-dir {dir_save} --remote True --n-epochs 1 --n-epochs-decay 1 --gpu-ids 0 --gpu-ids 1'
            if model_type == 'ext' and 'noseg' in dir_input:
                cmd += ' --seg-gen false'
            res = subprocess.run(cmd,shell=True)
            assert res.returncode == 0
    else:
        pytest.skip(f'Detected {available_gpus} (< 2) available GPUs. Skip.')