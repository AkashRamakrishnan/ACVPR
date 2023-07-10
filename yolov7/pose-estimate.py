import cv2
import time
import torch
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt,strip_optimizer,xyxy2xywh
from utils.plots import output_to_keypoint, plot_skeleton_kpts,colors,plot_one_box_kpt,plot_one_box_cogbox
import pandas as pd

# from filterpy.kalman import KalmanFilter
import json

@torch.no_grad()
def run(poseweights="yolov7-w6-pose.pt",source="football1.mp4",device='cpu',view_img=False,
        save_conf=False,line_thickness = 3,hide_labels=False, hide_conf=True):

    frame_count = 0  #count no of frames
    total_fps = 0  #count total fps
    time_list = []   #list to store time
    fps_list = []    #list to store fps
    id_dict = {}
    # Define Kalman filter parameters
    measurement_noise_x = 10  # Measurement noise in the x-direction
    measurement_noise_y = 10  # Measurement noise in the y-direction
    process_noise_x = 0.1  # Process noise in the x-direction
    process_noise_y = 0.1  # Process noise in the y-direction
    process_noise_vx = 0.1  # Process noise in the x-velocity
    process_noise_vy = 0.1  # Process noise in the y-velocity
    count = 0
    device = select_device(opt.device) #select device
    half = device.type != 'cpu'

    model = attempt_load(poseweights, map_location=device)  #Load model
    _ = model.eval()
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
   
    if source.isnumeric() :    
        cap = cv2.VideoCapture(int(source))    #pass video to videocapture object
    else :
        cap = cv2.VideoCapture(source)    #pass video to videocapture object
   
    if (cap.isOpened() == False):   #check if videocapture not opened
        print('Error while trying to read video. Please check path again')
        raise SystemExit()

    else:
        frame_width = int(cap.get(3))  #get video frame width
        frame_height = int(cap.get(4)) #get video frame height

        
        vid_write_image = letterbox(cap.read()[1], (frame_width), stride=64, auto=True)[0] #init videowriter
        resize_height, resize_width = vid_write_image.shape[:2]
        out_video_name = f"{source.split('/')[-1].split('.')[0]}"
        # out = cv2.VideoWriter(f"{source}_keypoint.mp4",
        #                     cv2.VideoWriter_fourcc(*'mp4v'), 30,
        #                     (resize_width, resize_height))
        kalman_filters = {}
        final_kpts = {}
        while(cap.isOpened): #loop until cap opened or video not complete
        
            print("Frame {} Processing".format(frame_count+1))

            ret, frame = cap.read()  #get frame and success from video capture
            
            if ret: #if success is true, means frame exist
                orig_image = frame #store frame
                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB) #convert frame to RGB
                image = letterbox(image, (frame_width), stride=64, auto=True)[0]
                image_ = image.copy()
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))
            
                image = image.to(device)  #convert image data to device
                image = image.float() #convert image to float precision (cpu)
                start_time = time.time() #start time for fps calculation
            
                with torch.no_grad():  #get predictions
                    output_data, _ = model(image)

                output_data = non_max_suppression_kpt(output_data,   #Apply non max suppression
                                            0.25,   # Conf. Threshold.
                                            0.65, # IoU Threshold.
                                            nc=model.yaml['nc'], # Number of classes.
                                            nkpt=model.yaml['nkpt'], # Number of keypoints.
                                            kpt_label=True)
            
                output = output_to_keypoint(output_data)

                im0 = image[0].permute(1, 2, 0) * 255 # Change format [b, c, h, w] to [h, w, c] for displaying the image.
                im0 = im0.cpu().numpy().astype(np.uint8)
                
                im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR) #reshape image format to (BGR)
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                #rects = []
                existing_ids = {}

                for i, pose in enumerate(output_data):  # detections per image
                
                    if len(output_data):  #check if no pose
                        for c in pose[:, 5].unique(): # Print results
                            n = (pose[:, 5] == c).sum()  # detections per class
                            print("No of Objects in Current Frame : {}".format(n))
                            print(type(pose))
                        counter = 0    
                        for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:,:6])): #loop over poses for drawing on frame
                            c = int(cls)
                            kpts = pose[det_index, 6:]
                            cx, cy = int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2)  # Calculate center of gravity
                            existing_ids[counter] = (cx, cy, kpts, xyxy)
                            counter += 1
                        
                print(existing_ids)            
                # disappeared_ids = []
                # matched_points1 = []
                # matched_points2 = []
                matched_indices1 = []
                matched_indices2 = []
                for bbox_id, (prev_cx, prev_cy, kpts, xyxy) in id_dict.items():
                    # dist = []
                    min_dist = 10000
                    min_index=None
                    for new_bbox_id, (cx, cy, kpts, xyxy) in existing_ids.items():
                      distance = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5
                      # dist.append(distance)
                      if distance < min_dist and distance <100:
                        if new_bbox_id not in matched_indices2:
                          min_dist = distance
                          min_index = new_bbox_id
                    if min_index is not None:
                      matched_indices1.append(bbox_id)
                      matched_indices2.append(min_index)
                plot_id = {}      
                for i in range(len(matched_indices1)):
                    bbox_info = existing_ids.get(matched_indices2[i])
                    if bbox_info is not None:
                        id_dict[matched_indices1[i]] = bbox_info
                        plot_id[matched_indices1[i]] = bbox_info
                        existing_ids.pop(matched_indices2[i])

                    # # print('distance')
                    # # print(dist)
                    # if dist:
                    #     min_val = min(dist)
                    #     min_id = dist.index(min_val)
                      
                    # if min_val < 100:  # Adjust the threshold as needed
                    #     # disappeared_ids.append(min_id)
                    #     bbox_info = existing_ids.get(min_id)  # Get bbox_info with min_id from existing_ids
                    #     if bbox_info is not None:
                    #         id_dict[bbox_id] = bbox_info
                    #         existing_ids.pop(min_id)
                    #       # id_dict[bbox_id] = existing_ids[min_id]
                    #       # existing_ids.pop(min_id)
                if existing_ids:
                  for bbox_id, (cx, cy, kpts, xyxy) in existing_ids.items():
                    id_dict[count] = (cx, cy, kpts, xyxy)
                    plot_id[count] = (cx, cy, kpts, xyxy)
                    count += 1

                for bbox_id, (cx, cy, kpts, xyxy) in plot_id.items():
                    text_id = f"ID: {bbox_id}"
                    print(text_id)
                    # print(kpts)
                    steps = 3
                    num_kpts = len(kpts) // steps
                    xy_kpts = [frame_count]
                    for kid in range(num_kpts):
                        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
                        xy_kpts.append(int(x_coord.item()))
                        xy_kpts.append(int(y_coord.item()))
                    plot_one_box_kpt(xyxy, text_id, im0, color=colors(c, True), line_thickness=line_thickness,
                                      kpt_label=True, kpts=kpts, steps=3, orig_shape=im0.shape[:2],
                                      predicted_cx=None, predicted_cy=None)
                    print(len(xy_kpts))
                    if bbox_id in final_kpts:
                        final_kpts[bbox_id].append(xy_kpts)
                    else:
                        final_kpts[bbox_id] = [xy_kpts]
                   
                end_time = time.time()  #Calculatio for FPS
                fps = 1 / (end_time - start_time)
                total_fps += fps
                frame_count += 1
                
                fps_list.append(total_fps) #append FPS in list
                time_list.append(end_time - start_time) #append time in list
                # Stream results
                if view_img:
                    cv2.imshow("YOLOv7 Pose Estimation Demo", im0)
                    cv2.waitKey(1)  # 1 millisecond

                # out.write(im0)  #writing the video frame

            else:
                break
        # print(final_kpts)
        cap.release()
        # cv2.destroyAllWindows()
        avg_fps = total_fps / frame_count
        print(f"Average FPS: {avg_fps:.3f}")
        # with open('shooting2.json', 'w') as f:
        #     json.dump(final_kpts, f)
        #plot the comparision graph
        plot_fps_time_comparision(time_list=time_list,fps_list=fps_list)
        #save csv files
        main_dir = '/home/s3075451/ACVPR/yolov7-pose-estimation/trajectories'
        vid_name = opt.source.split('/')[-1].split('.')[0]
        save_dir = os.path.join(main_dir, vid_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for key in final_kpts.keys():
            fname = str(key)+'.csv'
            # print(final_kpts[key])
            df = pd.DataFrame(final_kpts[key])
            # print(df)
            df.to_csv(os.path.join(save_dir, fname), index=False, header=False)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='football1.mp4', help='video/0 for webcam') #video source
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')   #device arugments
    parser.add_argument('--view-img', action='store_true', help='display results')  #display results
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels') #save confidence in txt writing
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)') #box linethickness
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels') #box hidelabel
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences') #boxhideconf
    opt = parser.parse_args()
    return opt

#function for plot fps and time comparision graph
def plot_fps_time_comparision(time_list,fps_list):
    plt.figure()
    plt.xlabel('Time (s)')
    plt.ylabel('FPS')
    plt.title('FPS and Time Comparision Graph')
    plt.plot(time_list, fps_list,'b',label="FPS & Time")
    plt.savefig("FPS_and_Time_Comparision_pose_estimate.png")
    

#main function
def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device,opt.poseweights)
    main(opt)
