# Object Detection Installation Guide 🚀

This guide will help you set up the Object Detection project on your machine using Anaconda. Follow these steps to get started!

## Prerequisites 📋

Ensure you have Anaconda installed on your machine. If not, download it here:

## Installation Steps 🛠️

1. **Install Anaconda**
   [Anaconda Download](https://www.anaconda.com/download/success)
   - Then launch the Anaconda Prompt by opening your start menu and looking for Anaconda Prompt

3. **Create and Activate the Python Environment**
   ```bash
   conda create -n detectron_env python=3.8 -y
   conda activate detectron_env
   conda install git -y
   ```

4. **Clone the Repository**
   ```bash
   git clone https://github.com/pentestfunctions/object-detection.git
   cd object-detection
   ```

5. **Install Dependencies**
   ```bash
   python setup.py
   ```

6. **Download the Model Dataset**
   - Go to the dataset page: [Model Dataset](https://universe.roboflow.com/navrachana-university-l5d92/car_models-izfw0/dataset/1)
   - Click "Download Dataset"
   - Choose "Download zip to computer" and select Format (COCO JSON)
   - Copy the downloaded zip file to the folder containing `main.py`

7. **Train the Model**
   ```bash
   python main.py --train
   ```
   This command finds any zip files in the folder with the train/verify folders, adds them to a menu for you to choose which one to train on, extracts them, and starts training. The classes count is automatically determined, so no need to worry!

## Live Detection 🎥

If you are using a virtual camera like OBS, make sure it is running as a virtual camera, then execute:
```bash
python main.py --live --cam 2
```
Change the `--cam` number to the number of the device you are using.

## Support or Contact 🤝

Having trouble with the setup? Open an issue in the [GitHub repository](https://github.com/pentestfunctions/object-detection/issues) for support.
