import os
import sys
import cv2
import glob
import json
import torch
import random
import zipfile
import argparse
import requests
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.utils.visualizer import Visualizer, ColorMode

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class Detector:
    def __init__(self, roi_num_classes, roi_threshold_test, skip_frames=5):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = "output/model_final.pth"
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = roi_num_classes
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = roi_threshold_test
        self.cfg.MODEL.DEVICE = device
        self.predictor = DefaultPredictor(self.cfg)
        self.metadata = MetadataCatalog.get("my_dataset_train")
        self.skip_frames = skip_frames

    def predictWebcam(self, webcam_device):
        capture = cv2.VideoCapture(webcam_device)
        if not capture.isOpened():
            print("Error: Webcam not accessible")
            return

        frame_count = 0
        last_predictions = None

        while True:
            ret, frame_original = capture.read()
            if not ret:
                break

            frame = cv2.resize(frame_original, (640, 480))
            if frame_count % self.skip_frames == 0:  # Predict on every Nth frame
                last_predictions = self.predictor(frame)
                print("Predictions:", last_predictions)  # Logging predictions for debug

            if last_predictions:
#                v = Visualizer(frame_original[:, :, ::-1], metadata=self.metadata, instance_mode=ColorMode.IMAGE)
                v = Visualizer(frame[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), instance_mode=ColorMode.IMAGE)
                output = v.draw_instance_predictions(last_predictions["instances"].to("cpu"))
                frame_original = output.get_image()[:, :, ::-1]

            cv2.imshow("Video", frame_original)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1

        capture.release()
        cv2.destroyAllWindows()

def ensure_config_file(sub_dir, cfg_filename, url):
    cfg_dir_path = os.path.join("detectron2", "configs", sub_dir)
    cfg_file_path = os.path.join(cfg_dir_path, cfg_filename)
    os.makedirs(cfg_dir_path, exist_ok=True)  # Ensure the directory exists
    if not os.path.exists(cfg_file_path):
        print(f"{cfg_filename} not found locally, downloading now from {url}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(cfg_file_path, 'wb') as f:
                f.write(response.content)
            print("Download completed")
        else:
            raise Exception(f"Failed to download {cfg_filename} from {url}")
    else:
        print(f"{cfg_filename} found locally....")

def register_dataset(folder_name):
    register_coco_instances("my_dataset_train", {}, f"{folder_name}/train/_annotations.coco.json", f"{folder_name}/train")
    register_coco_instances("my_dataset_test", {}, f"{folder_name}/test/_annotations.coco.json", f"{folder_name}/test")

def visualize_training_data():
    my_dataset_train_metadata = MetadataCatalog.get("my_dataset_train")
    dataset_dicts = DatasetCatalog.get("my_dataset_train")
    for idx, d in enumerate(random.sample(dataset_dicts, 3)):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_train_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imwrite(f"sample_image_{idx}.jpg", vis.get_image()[:, :, ::-1])

def train_custom_detectron2(roi_num_classes):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_cfg()

    # Directly specify the configuration file path
    config_file_path = os.path.join("detectron2", "configs", "COCO-Detection", "faster_rcnn_R_50_FPN_3x.yaml")
    
    if os.path.exists(config_file_path):
        cfg.merge_from_file(config_file_path)
    else:
        raise FileNotFoundError(f"Configuration file not found at {config_file_path}")

    cfg.MODEL.DEVICE = device
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.MAX_ITER = 500
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = roi_num_classes

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()


def clean_files():
    for file in glob.glob("README*"):
        os.remove(file)
    for file in glob.glob("sample_*.jpg"):
        os.remove(file)
    print("Cleanup complete.")

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def find_zip_files():
    current_directory = os.getcwd()
    all_files = os.listdir(current_directory)
    zip_files = [file for file in all_files if file.endswith('.zip')]
    
    matching_files = []

    for zip_file in zip_files:
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                first_level_contents = set()

                for item in zip_ref.namelist():
                    parts = item.split('/')
                    if len(parts) > 1 and parts[1] == '':
                        first_level_contents.add(parts[0])

                if 'train' in first_level_contents and 'valid' in first_level_contents:
                    matching_files.append(zip_file)
        except zipfile.BadZipFile:
            print(f"Failed to read {zip_file} as it is not a valid zip file.")
    
    if not matching_files:
        print("No matching zip files found.")
        return None
    
    print("Choose a file:")
    for i, file_name in enumerate(matching_files, start=1):
        print(f"{i}. {file_name}")
    
    while True:
        try:
            choice = int(input("Enter the number of the file you want to choose: "))
            if choice < 1 or choice > len(matching_files):
                raise ValueError
            chosen_file = matching_files[choice - 1]
            return chosen_file
        except ValueError:
            print("Invalid input. Please enter a valid number.")

def unzip_file_to_use(chosen_file):
    try:
        folder_name = os.path.splitext(chosen_file)[0]  # Remove the .zip extension
        
        # Check if the folder already exists
        if os.path.exists(folder_name):
            print(f"The folder '{folder_name}' already exists.")
            overwrite = input("Do you want to overwrite it? (y/n): ")
            if overwrite.lower() != 'y':
                print("Aborted. Please choose a different file or rename the existing folder.")
                return None
        
        with zipfile.ZipFile(chosen_file, 'r') as zip_ref:
            zip_ref.extractall(folder_name)
            return folder_name
    except zipfile.BadZipFile:
        print(f"Failed to read {chosen_file} as it is not a valid zip file.")
        return None

def load_roi_num_classes(file_path):
    """Load ROI num classes from a text file."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return int(file.read().strip())
    else:
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
def get_roi_num_classes(folder_path):
    """
    Reads class names from a JSON file, writes the number of unique classes to a text file,
    and returns the number of unique classes.

    Args:
    folder_path (str): The path to the folder containing the JSON file.

    Returns:
    int: The number of unique ROI classes.
    """
    json_file_path = os.path.join(folder_path, "train", "_annotations.coco.json")
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
            class_names = [category["name"] for category in data["categories"]]
            num_classes = len(set(class_names))
            roi_num_classes = num_classes

            output_dir = "output"
            output_path = os.path.join(output_dir, "classes_used.txt")
            os.makedirs(output_dir, exist_ok=True)

            with open(output_path, "w") as output:
                output.write(str(roi_num_classes))
                
            return roi_num_classes
    except FileNotFoundError:
        print(f"Error: The file {json_file_path} does not exist.")
        return None
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Detectron2 model or run live detection.')
    parser.add_argument('--train', action='store_true', help='Initiate training process')
    parser.add_argument('--live', action='store_true', help='Initiate webcam/virtual camera')
    parser.add_argument('--clean', action='store_true', help='Clean specific files from the directory')
    parser.add_argument('--cam', type=int, default=0, help='Device number to use (0 by default)')
    parser.add_argument('--folder', type=str, help='Path to the dataset folder')
    args = parser.parse_args()

    # Amount of items to classify from the training set
    roi_num_classes = 1

    # Sets the threshold for deciding whether detections are considered positive
    roi_threshold_test = 0.5

    ensure_config_file("COCO-Detection", "faster_rcnn_R_50_FPN_3x.yaml",
                       "https://raw.githubusercontent.com/facebookresearch/detectron2/main/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    ensure_config_file("", "Base-RCNN-FPN.yaml",
                       "https://raw.githubusercontent.com/facebookresearch/detectron2/main/configs/Base-RCNN-FPN.yaml")
    try:
        if args.clean:
            clean_files()
        elif args.train:
            if not args.folder:
                # Call the function to execute the code
                chosen_file = find_zip_files()
                print("Potential files found for training data - choose a file:", chosen_file)
                unzipped_folder = unzip_file_to_use(chosen_file)
                print("Unzipped folder:", unzipped_folder)
                args.folder = unzipped_folder
            if not os.path.exists(f"{args.folder}/train") and not os.path.exists(f"{args.folder}/valid"):
                print(f"The folder you specified didn't seem to contain a train and valid folder for training.")
                sys.exit()
            if os.path.exists("output/model_final.pth"):
                response = input("Model file already exists. Do you want to continue and overwrite it? (yes/no): ")
                if response.lower() != 'yes':
                    print("Training aborted.")
                    sys.exit()
                else:
                    for file in glob.glob("output/*"):
                        os.remove(file)
            clear_screen()
            roi_num_classes = get_roi_num_classes(args.folder)
            register_dataset(args.folder)
            visualize_training_data()
            train_custom_detectron2(roi_num_classes)
            print("Output file in: output/model_final.pth")
        elif args.live:
            clear_screen()
            classes_file_path = os.path.join('output', 'classes_used.txt')
            try:
                roi_num_classes = load_roi_num_classes(classes_file_path)
                detector = Detector(roi_num_classes, roi_threshold_test, skip_frames=60)
                detector.predictWebcam(args.cam)
            except FileNotFoundError as e:
                print(e)
            except ValueError:
                print("Error: The file does not contain a valid integer.")
        else:
            print("No action initiated. Use '--train' or '--live' with the appropriate options.")
    except FileNotFoundError as e:
        print(e)
