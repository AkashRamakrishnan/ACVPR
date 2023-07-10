import subprocess
import os

video_types = ["abuse", "assault", "shooting", "vandalism"]
base_path = "/deepstore/datasets/dmb/MachineLearning/HRC/HRC_files/UCF_Videos/"
pose_estimate_script = "/home/s3075451/ACVPR/yolov7-pose-estimation/pose-estimate.py"
# for video_type in video_types:
# for folder in os.listdir(base_path):
for folder in ['Shoplifting', 'Stealing']:
    # folder_path = os.path.join(base_path, video_type)
    folder_path = os.path.join(base_path, folder)
    # Loop through all video files in the folder
    for video_file in os.listdir(folder_path):
        video_path = os.path.join(folder_path, video_file)
        print("Processing video:", video_path)
        # Call pose-estimate.py with the necessary arguments using subprocess
        subprocess.run(["python", pose_estimate_script, "--source", video_path, "--device", "0"])
