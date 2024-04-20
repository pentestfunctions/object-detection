import subprocess
import os
import shutil

def get_cuda_version():
    """Fetches the CUDA version using nvidia-smi command and returns it."""
    # Check if nvidia-smi is available on the PATH
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
    if cuda_version.startswith('11'):
        run_command('conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y')
    elif cuda_version.startswith('12'):
        run_command('conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y')
    else:
        print("No compatible PyTorch installation for CUDA version - please change or install CUDA:", cuda_version)

def run_command(command, cwd=None):
    """Runs a shell command in an optional specific directory and prints the outcome."""
    try:
        subprocess.run(command, check=True, shell=True, cwd=cwd)
        print(f"Successfully executed: {command}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

def main():
    # Install necessary libraries
    run_command('pip install cython')

    # Fetch CUDA version
    cuda_version = get_cuda_version()
    if not cuda_version:
        # Exit the script if no CUDA version is detected
        return

    # Proceed with the installation if CUDA is detected
    install_pytorch(cuda_version)

    # Path to the directory where detectron2 should be
    detectron_dir = os.path.join(os.getcwd(), 'detectron2')
    
    if not os.path.exists(detectron_dir):
        # Clone detectron2 if the directory does not exist
        run_command('git clone https://github.com/facebookresearch/detectron2.git')
    else:
        print("Detectron2 directory already exists. Consider pulling updates or removing it.")

    # Install detectron2 in editable mode
    if os.path.exists(detectron_dir):
        run_command('pip install -e .', cwd=detectron_dir)

    # Install opencv-python
    run_command('pip install opencv-python')

if __name__ == "__main__":
    main()
