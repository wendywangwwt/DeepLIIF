import subprocess
import os
#from conftest import MODEL_INFO

# latest commit: '9b3604d'
COMMIT_INFO = {'latest':'tests-image-quality-baseline-latest', # 20230307 Merge pull request nadeemlab#25 from wendywangwwt/main-stable-inference
               'ext':'c1e7aee',#20231222 Use 1024x1024 for deepliif extension model #'tests-image-quality-baseline-ext', #20230725 Merge pull request nadeemlab#30 from wendywangwwt/main-ext-model-class
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
    flag_error = False
    msg_error = ''
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
        flag_error = True
        msg_error += str(e)
    
    finally:
        # switch back to the original branch/commit
        print(f'Switch back to {current_branch}...')
        subprocess.check_call(f"git checkout {current_branch}", shell=True, stdout=subprocess.DEVNULL)
        
        # pop stash if needed
        if "No local changes" not in stash_output:
            print('Pop stash..')
            subprocess.check_call("git stash pop", shell=True, stdout=subprocess.DEVNULL)
        
        if flag_error:
            raise Exception(f'Failed: {msg_error}')


def ensure_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)


def generate_baseline(commit_info, model_info, inference_type=['testpy','cli'], index=None, verbose=0):
    """
    The main function to generate baseline inference output to be compared against.
    
    commit_info: a dictionary with model type as key and commit hash as value. Default to COMMIT_INFO specified in this script.
                 otherwise, a string of commit hash
    model_info: model info as used in other test scripts that stores metadata
    inference_type: a list with elements in ['testpy', 'cli'], only the chosen inference types will be run
    index: for model info with multiple candidates (e.g., ext), index controls which candidate to generate output for; None will use 
           all candidates
    """
    # Run a command at a specific commit
    if commit_info is None:
        commit_info = COMMIT_INFO
    if isinstance(commit_info,str):
        commit_info = {model_info['model']:commit_info}
        
    for model_type,commit_hash in commit_info.items():
        if verbose > 0:
            print('-'*50)
            print(f'Generating standard output for model type/name {model_type}...')
        #model_info = MODEL_INFO[model_type]
        model_name = model_info['model']
        dirs_model = model_info['dir_model']
        tile_size = model_info['tile_size']
        
        # 1. testpy
        if 'testpy' in inference_type:
            dirs_input = model_info['dir_input_testpy']
            dirs_output = model_info['dir_output_baseline_testpy']
            for i, (dir_model, dir_input, dir_output) in enumerate(zip(dirs_model, dirs_input, dirs_output)):
                if index is None or index == i:
                    if verbose > 0:
                        print(f'Running testpy inference with model {dir_model}...')
                    ensure_exists(dir_output)
                    
                    if model_name in ['DeepLIIF','DeepLIIFExt']:
                        # deepliif model at that time still requires --model and --name as input to test.py
                        cmd = f"python test.py --checkpoints_dir {dir_model} --dataroot {dir_input} --model {model_name} --name . --results_dir {dir_output}"
                        res = run_command_at_git_version(commit_hash,cmd)
                        if res.returncode != 0:
                            cmd = f"python test.py --checkpoints_dir {dir_model} --dataroot {dir_input} --name . --results_dir {dir_output}"
                            res = run_command_at_git_version(commit_hash,cmd)
                        
                        if res.returncode != 0:
                            raise Exception(f'FAILED: {res.stderr}')
                    else:
                        cmd = f"python test.py --checkpoints_dir {dir_model} --dataroot {dir_input} --results_dir {dir_output}"
                        res = run_command_at_git_version(commit_hash,cmd)
                    if res.returncode != 0:
                        raise Exception(f'FAILED: {res.stderr}')
            if verbose > 0:
                print(f'Finished testpy inference for model type {model_type}.')
                print('*'*50)
        
        # 2. cli inference
        if 'cli' in inference_type:
            dirs_input = model_info['dir_input_inference']
            dirs_output = model_info['dir_output_baseline_inference']        
            for i, (dir_model, dir_input, dir_output) in enumerate(zip(dirs_model, dirs_input, dirs_output)):
                if index is None or index == i:
                    if verbose > 0:
                        print(f'Running cli inference with model {dir_model}...')
                    ensure_exists(dir_output)
                    cmd = f'python cli.py test --model-dir {dir_model} --input-dir {dir_input} --output-dir {dir_output} --tile-size {tile_size}'
                    res = run_command_at_git_version(commit_hash,cmd)
                    if res.returncode != 0:
                        raise Exception(f'FAILED: {res.stderr}')
            if verbose > 0:
                print(f'Finished cli inference for model type/name {model_type}.')
    print('Completed generating standard output for all model types.')

    