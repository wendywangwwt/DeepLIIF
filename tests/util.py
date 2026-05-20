from skimage.metrics import structural_similarity as ssim
from PIL import Image
import cv2
import numpy as np
import shutil
import os
import pynvml
import subprocess
import time
import torch

def calculate_ssim(dir_a,dir_b,fns,suffix_a,suffix_b,verbose_freq=50):
    print(suffix_a, suffix_b)
    score = 0
    count = 0
    
    for fn in fns:
        path_a = f"{dir_a}/{fn}{suffix_a}.png"
        assert os.path.exists(path_a), f'path {path_a} does not exist'
        img_a = cv2.imread(path_a)
        # print(img_a.shape)
        
        if suffix_b == 'combine': 
            paths_b = [f"{dir_b}/{fn}fake_BS_{i}.png" for i in range(1,5)]
            imgs_b = []
            for path_b in paths_b:
                assert os.path.exists(path_b), f'path {path_b} does not exist'
                imgs_b.append(cv2.imread(path_b))
            img_b = np.mean(imgs_b,axis=(0))
        else:
            path_b = f"{dir_b}/{fn}{suffix_b}.png"
            assert os.path.exists(path_b), f'path {path_b} does not exist'
            img_b = cv2.imread(path_b)
        # print(img_b.shape)
        
        score += ssim(img_a,img_b, data_range=img_a.max() - img_b.min(), multichannel=True, channel_axis=2)
        count +=1

        #if count % verbose_freq == 0 or count == len(fns):
        #    print(f"{count}/{len(fns)}, running mean SSIM {score / count}")
    return score/count


# https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder
def remove_contents_in_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def check_gpus_memory(l_gpu_ids_to_check):
    """
    In order to check if the specified gpus are in use as expected,
    we check if the gpu memory usage on the speicifed gpu devices are 
    increased after the process starts.
    
    This function returns the gpu memory usage for each given device.
    
    Note that the test gpus should have no other processes running,
    otherwise the increase in memory could be from other processes
    rather than the process we expect. Or, the other running processes
    should have very stable memory consumption which can be used as
    a reliable baseline.
    """
    # get baseline
    pynvml.nvmlInit()
    d_memory_usage = {gpu_id: pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(gpu_id)).used
                      for gpu_id in l_gpu_ids_to_check}
    pynvml.nvmlShutdown()
    
    return d_memory_usage


def get_gpus_for_pid(pid):
    """
    Gather all GPU IDs on which a given process id is running.
    
    Note that this does not work if run from within a container.
    In this case, the pid produced by subprocess is from within docker 
    while the pid captured by nvml through gpu driver is from the host 
    machine. They MISMATCH, and we cannot map one to the other.
    """
    pynvml.nvmlInit()
    
    l_gpu_ids = []
    device_count = pynvml.nvmlDeviceGetCount()

    for gpu_id in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        if any(p.pid == pid for p in processes):
            l_gpu_ids.append(gpu_id)
    
    pynvml.nvmlShutdown()
    return l_gpu_ids
    

def run_subprocess_and_check_device(cmd, l_gpu_ids_to_check=[0], check_gpu_with_pid=True, gpu_in_use=True):
    """
    Run a subprocess command within a test case and check gpu usage.
    We first check a set of 2x2 conditions: 
      - whether there is gpu device (available_gpus > 0)
      - whether gpu usage is expected (gpu_in_use = True)
    Then the other set of 2x2 conditions:
      - whether use pid to pull gpu device usage (check_gpu_with_pid = True) or gpu memory consumption
      - whether gpu usage is expected (gpu_in_use = True)
    
    cmd: the basic command
    l_gpu_ids_to_check: check the listed gpu ids and see whether they are used; if flag
        gpu_in_use is True, appends --gpu-ids args to the command string
    check_gpu_with_pid: whether we determine gpu usage by matching running process id on
        selected gpu devices (True, calls get_gpus_for_pid) or by checking increase in gpu memory
        consumption (False, calls check_gpus_memory)
    gpu_in_use: if True, check whether the selected gpus are in use; if False, check whether they
        are NOT in use (to test cpu computation)
    """
    available_gpus = torch.cuda.device_count()
        
    if gpu_in_use:
        assert available_gpus > 0, f'no available gpu to run test case {cmd}' # 1. check if gpu is used when no gpu available - fail
        l_gpu_ids_all = list(range(available_gpus))
        for gpu_id in l_gpu_ids_to_check:
            assert gpu_id in l_gpu_ids_all, f'gpu id {gpu_id} is not available (available gpus {l_gpu_ids_all})'
            cmd += f' --gpu-ids {gpu_id}'
    else:
        if available_gpus > 0:
            l_gpu_ids_to_check = list(range(torch.cuda.device_count()))
    
    if not gpu_in_use and available_gpus == 0: # 2. check if gpu is not used when no gpu available - always the case, no need to prob gpu devices
        process = subprocess.Popen(cmd,shell=True)
        return process.wait()
    else: # 3. check if gpu is (not) used when gpus are available
        if not check_gpu_with_pid:
            d_memory_usage_before = check_gpus_memory(l_gpu_ids_to_check)
            d_memory_usage_max  = {gpu_id:0 for gpu_id in d_memory_usage_before.keys()}
        else:
            l_gpu_ids_effective = []
        process = subprocess.Popen(cmd,shell=True)
        
        while process.poll() is None: # check return code, if not finished .poll() returns None
            if check_gpu_with_pid:
                if gpu_in_use: # if the command should trigger gpus, as long as we see all expected gpus got the pid, we stop probing
                    if sorted(l_gpu_ids_to_check) != sorted(l_gpu_ids_effective):
                        l_gpu_ids_effective = get_gpus_for_pid(process.pid)
                    else:
                        break
                else: # if the command should run only on cpu, as long as we see one gpu got the pid, we stop probing as this is wrong
                    l_gpu_ids_effective = get_gpus_for_pid(process.pid)
                    assert len(l_gpu_ids_effective) == 0, f'command {cmd} should not use GPU, but GPU devices ({l_gpu_ids_effective}) got the process on them'
            else:
                d_memory_usage_current = check_gpus_memory(l_gpu_ids_to_check)
                for gpu_id, usage in check_gpus_memory(l_gpu_ids_to_check).items():
                    if d_memory_usage_max[gpu_id] < usage:
                        d_memory_usage_max[gpu_id] = usage
            time.sleep(1)
        
        if check_gpu_with_pid and gpu_in_use:
            assert sorted(l_gpu_ids_to_check) == sorted(l_gpu_ids_effective), f'detected gpus used by the process ({l_gpu_ids_effective}) is different from the expected ones ({l_gpu_ids_to_check})'
        elif not check_gpu_with_pid and gpu_in_use:
            for gpu_id in l_gpu_ids_to_check:
                assert d_memory_usage_max[gpu_id] > d_memory_usage_before[gpu_id], f'gpu id {gpu_id} should but does not show increase in memory consumption'
        elif not check_gpu_with_pid and not gpu_in_use:
            for gpu_id in l_gpu_ids_to_check:
                assert d_memory_usage_max[gpu_id] <= d_memory_usage_before[gpu_id], f'gpu id {gpu_id} should not but does show increase in memory consumption'
        else: # check_gpu_with_pid and not gpu_in_use - the previous assert in while loop handles this
            pass
        
        return process.wait()


def run_function_and_check_device(func, kwargs={}, l_gpu_ids_to_check=[0], gpu_in_use=True):
    """
    Run a function and check gpu usage using torch's internal tracker.
    We first check a set of 2x2 conditions: 
      - whether there is gpu device (available_gpus > 0)
      - whether gpu usage is expected (gpu_in_use = True)
    Then 2 conditions:
      - whether gpu usage is expected (gpu_in_use = True)
    
    func: the basic function
    kwargs: a list of arg name and value passed to func
    l_gpu_ids_to_check: check the listed gpu ids and see whether they are used; if flag
        gpu_in_use is True, appends {"gpu_ids":l_gpu_ids_to_check} to func's opt_args; otherwise
        appends {"gpu_ids":[]}
    gpu_in_use: if True, check whether the selected gpus are in use; if False, check whether they
        are NOT in use (to test cpu computation)
    """
    available_gpus = torch.cuda.device_count()
    
    if 'opt_args' not in kwargs.keys():
        kwargs['opt_args'] = {}
    
    if gpu_in_use:
        assert available_gpus > 0, f'no available gpu to run test case {func.__name__}' # 1. check if gpu is used when no gpu available - fail
        l_gpu_ids_all = list(range(available_gpus))
        kwargs['opt_args']['gpu_ids'] = []
        for gpu_id in l_gpu_ids_to_check:
            assert gpu_id in l_gpu_ids_all, f'gpu id {gpu_id} is not available (available gpus {l_gpu_ids_all})'
            kwargs['opt_args']['gpu_ids'].append(gpu_id)
    else:
        if available_gpus > 0:
            kwargs['opt_args']['gpu_ids'] = []
            l_gpu_ids_to_check = list(range(torch.cuda.device_count()))
            print(f'Check gpu ids {l_gpu_ids_to_check} and ensure none is used')
    
    if not gpu_in_use and available_gpus == 0: # 2. check if gpu is not used when no gpu available - always the case, no need to prob gpu devices
        return func(**kwargs)
    else: # 3. check if gpu is (not) used when gpus are available
        # record baseline memory before resetting peak stats 
        d_memory_before = {}
        for gpu_id in l_gpu_ids_to_check:
            torch.cuda.set_device(gpu_id)
            d_memory_before[gpu_id] = torch.cuda.memory_allocated(gpu_id)
            
        # reset memory stats
        for gpu_id in l_gpu_ids_to_check:
            torch.cuda.set_device(gpu_id) # initialize cuda context
            torch.cuda.reset_peak_memory_stats(gpu_id) # # collect memory stats through cuda
        
        res = func(**kwargs)
        
        d_memory_usage_max = {gpu_id: torch.cuda.max_memory_allocated(gpu_id) for gpu_id in l_gpu_ids_to_check}
        
        if gpu_in_use:
            for gpu_id in l_gpu_ids_to_check:
                assert d_memory_usage_max[gpu_id] > d_memory_before[gpu_id], f'gpu id {gpu_id} should but does not show increase in memory consumption ({d_memory_usage_max[gpu_id]}); baseline: {d_memory_before[gpu_id]}'
        else:
            for gpu_id in l_gpu_ids_to_check:
                assert d_memory_usage_max[gpu_id] <= d_memory_before[gpu_id], f'gpu id {gpu_id} should not but does show increase in memory consumption ({d_memory_usage_max[gpu_id]}); baseline: {d_memory_before[gpu_id]}' # tolerance: 300MB if needed
