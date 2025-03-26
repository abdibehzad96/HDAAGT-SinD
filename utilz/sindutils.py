# from DataReader import *
import os
import torch
from torch.utils.data import Dataset, DataLoader
import yaml
import re
from utils import *


if __name__ == '__main__':
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    NUsers = 64
    NFeatures = 14
    downsample_sl = 2
    downsample_f = 6
    sl = 60
    future = 90
    sl2 = sl//2
    tot_len = sl + future
    Scene_clss = Scenes(sl//downsample_sl, future//downsample_f, NUsers, NFeatures, device)
    scene = torch.zeros(tot_len,NUsers,NFeatures, device=device)
    global_list = torch.zeros(NUsers)
    # path = '/home/abdikhab/New_Idea_Traj_Pred/sinD/Data/Chongqing/6_22_NR_1'
    # lightpath = '/home/abdikhab/New_Idea_Traj_Pred/sinD/Data/Chongqing/6_22_NR_1/TrafficLight_06_22_NR1_add_plight.csv'
    path = 'data/Changchun/'
    lightpath = 'data/Changchun/Traffic_Lights.csv'
    

    Veh_tracks_dict, Ped_tracks_dict = read_tracks_all(path)
    max_frame = 0
    min_frame = 1000
    for _, track in Veh_tracks_dict.items():
        frame= torch.from_numpy(track["frame_id"])
        frame_min = frame.min()
        frame_max = frame.max()
        max_frame = frame_max if frame_max > max_frame else max_frame
        min_frame = frame_min if frame_min < min_frame else min_frame
    print("Max frame",max_frame, "Min frame", min_frame)

    _, light = read_light(lightpath, max_frame)
    ZoneConfs = read_zones("utilz/ZoneConf.yaml")
    Boundaries = torch.load("Pickled/Boxes.pth",weights_only=True)
    Boundaries = torch.tensor(Boundaries).cpu()
    more_list = []
    for Frme in range(min_frame,max_frame, 5):
        scene = torch.zeros(tot_len,NUsers,NFeatures, device=device)
        adj_mat = torch.zeros(sl,NUsers,NUsers, device=device)
        global_list = {}
        order = 0
        last_Frame = Frme + sl + future
        more = 0
        for _, track in Veh_tracks_dict.items():
            id = torch.tensor(track["track_id"])
            frame= torch.from_numpy(track["frame_id"])
            x = torch.from_numpy(track["x"])
            y = torch.from_numpy(track["y"])
            vx = torch.from_numpy(track["vx"])
            vy = torch.from_numpy(track["vy"])
            ax = torch.from_numpy(track["ax"])
            ay = torch.from_numpy(track["ay"])
            v_lon = torch.from_numpy(track["v_lon"])
            v_lat = torch.from_numpy(track["v_lat"])
            a_lon = torch.from_numpy(track["a_lon"])
            a_lat = torch.from_numpy(track["a_lat"])
            yaw_rad = torch.from_numpy(track["yaw_rad"])
            heading_rad = torch.from_numpy(track["heading_rad"])
            if last_Frame in frame:
                for fr in range(sl2):
                    real_frame = fr+Frme
                    if real_frame in frame and id not in global_list:
                        indices = (frame >= real_frame) * (frame < last_Frame)
                        st_indx = torch.where(frame == real_frame)
                        end_indx = torch.where(frame == last_Frame)
                        if order < NUsers:
                            global_list[id] = order
                            ll = light[last_Frame-sum(indices):last_Frame]
                            scene[fr: tot_len,order] = torch.stack([x[indices],y[indices],vx[indices],vy[indices],yaw_rad[indices],ax[indices],ay[indices],a_lon[indices],a_lat[indices],
                                                                    v_lon[indices],v_lat[indices],heading_rad[indices],ll[:,0], ll[:,1]], dim=1)
                            order +=1
                            break
                        else:
                            more += 1
        adj_mat[:,:order, :order] = torch.eye(order, order, device=device).unsqueeze(0).repeat(sl, 1, 1)
        # occmap = occupancyMap(scene[-1], NUsers, Boundaries)
        # occmap = fast_occmap(scene[-1], NUsers)
        Scene_clss.add(scene[:sl:downsample_sl], scene[sl::downsample_f], adj_mat[0:sl:downsample_sl])
        more_list.append(more)
        


 
    final_path = 'data/Changchun/'
    Scene_clss.save(final_path)
    print("done; total length", Scene_clss.Scene.shape)


