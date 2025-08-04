import matplotlib.pyplot as plt
import numpy as np
from torchvision.io import write_video
import cv2
import os

# this script will create a dataset for the LAM model
# it will be a movie of very small size
# action controlling the location of one pixel
# and some noise in the last row
# 4x actions moving the pixel in the 4 directions

# size of the movie
n_pixels = (3, 4)
n_frames = 200
n_traj = 4000
n_noise_rows = 1
intensity = 0.25 # noise intensity
# save_path_data = '/home/timpearce/02_lam/dataset_01/'
# save_path_data = '/home/timpearce/02_lam/dataset_0.95/'
save_path_data = 'dataset/noise_0.25/'
save_path_viz = 'visualization/noise_0.25/'
policy_corr = 0

os.makedirs(save_path_data, exist_ok=True)
os.makedirs(save_path_viz, exist_ok=True)

# create the trajectory
for traj_i in range(n_traj):
    x = np.zeros((n_frames, n_pixels[0]+n_noise_rows, n_pixels[1]))
    a = np.zeros((n_frames, 1))
    a[:, 0] = np.random.randint(0, 5, n_frames) # random policy
    
    # init the pixel
    curr_pos = (n_pixels[0]//2, n_pixels[1]//2)
    x[0, curr_pos[0], curr_pos[1]] = 1

    for frame_i in range(n_frames-1):

        # if in a top corner, 90% of time take special action back to the center
        # if curr_pos[0] == 0 and curr_pos[1] == 0:
        #     if np.random.rand() < policy_corr:
        #         a[frame_i, 0] = 5
        # if curr_pos[0] == 0 and curr_pos[1] == 3:
        #     if np.random.rand() < policy_corr:
        #         a[frame_i, 0] = 5
        # if curr_pos[0] == 2 and curr_pos[1] == 0:
        #     if np.random.rand() < policy_corr:
        #         a[frame_i, 0] = 5
        # if curr_pos[0] == 2 and curr_pos[1] == 3:
        #     if np.random.rand() < policy_corr:
        #         a[frame_i, 0] = 5
        
        # snake pattern deterministic policy
        if False:
            if np.random.rand() < policy_corr:  # take deterministic action with high prob
                if curr_pos[0] == 0 and curr_pos[1] == 0: # row, column top left
                    a[frame_i, 0] = 1 # move down
                elif curr_pos[0] == 1 and curr_pos[1] == 0:
                    a[frame_i, 0] = 1
                elif curr_pos[0] == 2 and curr_pos[1] == 0:
                    a[frame_i, 0] = 3 # move right
                elif curr_pos[0] == 0 and curr_pos[1] == 1:
                    a[frame_i, 0] = 3
                elif curr_pos[0] == 1 and curr_pos[1] == 1:
                    a[frame_i, 0] = 0
                elif curr_pos[0] == 2 and curr_pos[1] == 1:
                    a[frame_i, 0] = 0 # move right
                elif curr_pos[0] == 0 and curr_pos[1] == 2:
                    a[frame_i, 0] = 1
                elif curr_pos[0] == 1 and curr_pos[1] == 2:
                    a[frame_i, 0] = 1
                elif curr_pos[0] == 2 and curr_pos[1] == 2:
                    a[frame_i, 0] = 3 # move right
                elif curr_pos[0] == 0 and curr_pos[1] == 3:
                    a[frame_i, 0] = 5
                elif curr_pos[0] == 1 and curr_pos[1] == 3:
                    a[frame_i, 0] = 0
                elif curr_pos[0] == 2 and curr_pos[1] == 3:
                    a[frame_i, 0] = 0 # move to top left

        a_i = a[frame_i, 0]
        if a_i == 0: # up
            curr_pos = (max(curr_pos[0] - 1,0), curr_pos[1])
        elif a_i == 1: # down
            curr_pos = (min(curr_pos[0] + 1,n_pixels[0]-1), curr_pos[1])
        elif a_i == 2: # left
            curr_pos = (curr_pos[0], max(curr_pos[1] - 1,0))
        elif a_i == 3: # right
            curr_pos = (curr_pos[0], min(curr_pos[1] + 1,n_pixels[1]-1))
        elif a_i == 4:  # static
            curr_pos = (curr_pos[0], curr_pos[1])
        elif a_i == 5:
            # special action to top left
            curr_pos = (0, 0)
            a[frame_i, 0] = 4 # overwrite this action
        # elif a_i == 5:
        #     # special action to return to the center
        #     curr_pos = (n_pixels[0]//2, n_pixels[1]//2)
        x[frame_i+1, curr_pos[0], curr_pos[1]] = 1
    
    # add noise
    x[:, n_pixels[0]:, :] = np.random.rand(n_frames, n_noise_rows, n_pixels[1]) < 0.5
    x[:, n_pixels[0]:, :] = x[:, n_pixels[0]:, :] * intensity # reduce intensity of the noise

    # save the trajectory
    np.save(save_path_data + 'x_'+str(traj_i)+'.npy', x)
    np.save(save_path_data + 'a_'+str(traj_i)+'.npy', a)

    # save as mp4 movie for visualization
    x_rgb = np.repeat(x[:, :, :, np.newaxis], 3, axis=3)
    x_rgb = 1 - x_rgb
    x_rgb = np.clip(x_rgb-0.1, 0, 1)
    x_rgb = x_rgb * 255
    x_rgb = x_rgb.astype(np.uint8)

    if traj_i <= 2:
        # filename_i = save_path_viz + 'vid_x_'+str(traj_i)+'.mp4'
        # write_video(filename_i, x_rgb, fps=10, video_codec="h264")
        # save as lossless avi
        filename_i = save_path_viz + 'vid_x_'+str(traj_i)+'.avi'
        # (200, 4, 4, 3)
        # upsample to 200x200
        # x_rgb = cv2.resize(x_rgb, (200, 200), interpolation=cv2.INTER_NEAREST)

        n_frames, height, width, channels = x_rgb.shape
        resized_frames = []
        for frame in x_rgb:
            resized_frame = cv2.resize(frame, (100, 100), interpolation=cv2.INTER_NEAREST)
            resized_frames.append(resized_frame)
        x_rgb_resized = np.array(resized_frames)

        write_video(filename_i, x_rgb_resized, fps=10, video_codec="h264", options={"crf": "0"})
    





