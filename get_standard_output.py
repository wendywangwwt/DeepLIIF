import subprocess
import os
from conftest import MODEL_INFO

COMMIT_INFO = {'latest':'tests-image-quality-baseline-latest', # 20230307 Merge pull request nadeemlab#25 from wendywangwwt/main-stable-inference
               'ext':'tests-image-quality-baseline-ext', #20230725 Merge pull request nadeemlab#30 from wendywangwwt/main-ext-model-class
               'sdg':'tests-image-quality-baseline-sdg', #20240207 Merge pull request nadeemlab#37 from wendywangwwt/main-sdg
               'cyclegan':'tests-image-quality-baseline-cyclegan' #20250402 Merge pull request nadeemlab#53 from wendywangwwt/main-pas-transfer
}

def run_command_at_git_version(commit_hash, command):
    """
    Switch to a specific git version and run a command using shell mode.
    
    Args:
        commit_hash: The git commit hash or tag to switch to
        command: Command to run (as string)
    """    
    try:
        # get the current branch/commit
        current_branch = subprocess.check_output("git rev-parse --abbrev-ref HEAD", shell=True).decode().strip()
        
        # stash uncommitted changes if any and keep the output to check later
        stash_output = subprocess.check_output("git stash", shell=True).decode().strip()
        
        # switch to specified commit
        subprocess.check_call(f"git checkout {commit_hash}", shell=True)

        # run key command
        res = subprocess.run(command, shell=True, capture_output=True, text=True)
        return res
        
    except Exception as e:
        print(f'Failed: {e}')
    
    finally:
        # switch back to the original branch/commit
        print(f'Switch back to {current_branch}...')
        subprocess.check_call(f"git checkout {current_branch}", shell=True, stdout=subprocess.DEVNULL)
        
        # pop stash if needed
        if "No local changes" not in stash_output:
            print('Pop stash..')
            subprocess.check_call("git stash pop", shell=True, stdout=subprocess.DEVNULL)

def ensure_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)

if __name__ == "__main__":
    # Run a command at a specific commit
    for model_type,commit_hash in COMMIT_INFO.items():
        print('-'*50)
        print(f'Generating standard output for model type {model_type}...')
        model_info = MODEL_INFO[model_type]
        model_name = model_info['model']
        dirs_model = model_info['dir_model']
        tile_size = model_info['tile_size']
        
        # 1. testpy
        dirs_input = model_info['dir_input_testpy']
        dirs_output = model_info['dir_output_standard_testpy']
        for dir_model, dir_input, dir_output in zip(dirs_model, dirs_input, dirs_output):
            print(f'Running testpy inference with model {dir_model}...')
            ensure_exists(dir_output)
            
            if model_name in ['DeepLIIF','DeepLIIFExt']:
                # deepliif model at that time still requires --model and --name as input to test.py
                cmd = f"python test.py --checkpoints_dir {dir_model} --dataroot {dir_input} --model {model_info['model']} --name . --results_dir {dir_output}"
                res = run_command_at_git_version(commit_hash,cmd)
            else:
                cmd = f"python test.py --checkpoints_dir {dir_model} --dataroot {dir_input} --results_dir {dir_output}"
                res = run_command_at_git_version(commit_hash,cmd)
            if res.returncode != 0:
                print('FAILED:',res.stderr)
        print(f'Finished testpy inference for model type {model_type}.')
        print('*'*50)
        
        # 2. cli inference
        dirs_input = model_info['dir_input_inference']
        dirs_output = model_info['dir_output_standard_inference']        
        for dir_model, dir_input, dir_output in zip(dirs_model, dirs_input, dirs_output):
            print(f'Running cli inference with model {dir_model}...')
            ensure_exists(dir_output)
            cmd = f'python cli.py test --model-dir {dir_model} --input-dir {dir_input} --output-dir {dir_output} --tile-size {tile_size}'
            res = run_command_at_git_version(commit_hash,cmd)
            if res.returncode != 0:
                print('FAILED:',res.stderr)
        print(f'Finished cli inference for model type {model_type}.')
    print('Completed generating standard output for all model types.')
