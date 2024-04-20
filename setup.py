import subprocess
import os
import shutil

def get_cuda_version():
    """Fetches the CUDA version using nvidia-smi command and returns it."""
    if shutil.which("nvidia-smi") is None:
        print("nvidia-smi not found. Ensure that NVIDIA drivers are installed and that nvidia-smi is in your PATH.")
        return None

    try:
        output = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        lines = output.stdout.split('\n')
        for line in lines:
            if 'CUDA Version' in line:
                parts = line.split('CUDA Version:')
                if len(parts) > 1:
                    cuda_version = parts[1].strip().split(' ')[0]
                    print("CUDA Version:", cuda_version)
                    return cuda_version
    except Exception as e:
        print("Failed to run nvidia-smi:", str(e))
    print("No CUDA version detected, make sure you have CUDA installed for your graphics card.")
    return None

def install_pytorch(cuda_version):
    """Installs PyTorch based on the CUDA version."""
    if cuda_version and cuda_version.startswith('11'):
        run_command('conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y')
    elif cuda_version and cuda_version.startswith('12'):
        run_command('conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y')
    else:
        print("No compatible PyTorch installation for CUDA version - please change or install CUDA:", cuda_version)
        install_cpu_only_packages()

def install_cpu_only_packages():
    """Installs necessary packages when no CUDA is available."""
    print("Installing CPU-only packages...")
    packages = "torch fvcore cloudpickle omegaconf torchvision pycocotools opencv-python cython"
    run_command(f"pip install {packages}")

def run_command(command, cwd=None):
    """Runs a shell command in an optional specific directory and prints the outcome."""
    try:
        subprocess.run(command, check=True, shell=True, cwd=cwd)
        print(f"Successfully executed: {command}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

def clean_project(base_dir='detectron2'):
    files_and_dirs = [
        ".circleci", ".github", "build", "configs", "datasets", "demo",
        "detectron2.egg-info", "dev", "docker", "docs", "projects", "tests",
        "tools", ".clang-format", ".flake8", ".gitignore", "GETTING_STARTED.md",
        "INSTALL.md", "LICENSE", "MODEL_ZOO.md", "setup.cfg"
    ]
    
    full_paths = [os.path.join(base_dir, item) for item in files_and_dirs]
    
    for item_path in full_paths:
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
            print(f"Removed directory: {item_path}")
        elif os.path.isfile(item_path):
            os.remove(item_path)
            print(f"Removed file: {item_path}")
        else:
            print(f"Item not found: {item_path}")

def main():
    run_command('pip install cython')

    cuda_version = get_cuda_version()
    if cuda_version:
        install_pytorch(cuda_version)
    else:
        install_cpu_only_packages()

    detectron_dir = os.path.join(os.getcwd(), 'detectron2')
    
    if not os.path.exists(detectron_dir):
        run_command('git clone https://github.com/facebookresearch/detectron2.git')
    
    main_py_path = os.path.join(os.getcwd(), 'main.py')
    detectron_dir_path = os.path.join(detectron_dir, 'main.py')
    if os.path.exists(main_py_path):
        shutil.move(main_py_path, detectron_dir_path)
        print(f"Moved main.py to {detectron_dir_path}")

    if os.path.exists(detectron_dir):
        run_command('pip install -e .', cwd=detectron_dir)

    run_command('pip install opencv-python')

if __name__ == "__main__":
    main()
    clean_project()
