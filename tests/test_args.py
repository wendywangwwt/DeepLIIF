import torch
from cpuinfo import get_cpu_info

def test_print_argument(model_type,model_dir):
    print("Displaying model_type: %s" % model_type)
    print("Displaying model_dir: %s" % model_dir)
    print("********** Hardware Info *********")
    print('GPU:')
    print(torch.cuda.get_device_name(),torch.cuda.device_count())
    cpu_info = get_cpu_info()
    del cpu_info['flags']
    print('CPU:')
    print(cpu_info)
    print('********** Hardware Info End *********')
