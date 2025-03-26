# This file will generate dataset for Scene centric models
from torch.utils.data import Dataset, DataLoader
import csv
import datetime
import os
import torch
import pandas as pd
import yaml
import re
from shapely.geometry import Point, Polygon
import numpy as np


class Scenes(Dataset):
    def __init__(self, config): # Features include ['BBX','BBY','W', 'L' , 'Cls', 'Xreal', 'Yreal']
        self.device = config['device']
        self.path = config['detection_path']
        self.Scene = torch.empty(0, config['sl']//config['dwn_smple'], config['Nnodes'], config['input_size'], device=self.device) #[Sampeles, Sequence Length, Nnodes (30), NFeatures(tba)]
        self.Adj_Mat = torch.empty(0, config['sl']//config['dwn_smple'], config['Nnodes'], config['Nnodes'], device=self.device) #[Sampeles, Sequence Length, Nnodes, Nnodes]
        self.Target = torch.empty(0, config['future']//config['dwn_smple'], config['Nnodes'], config['input_size'], device=self.device) #[Sampeles, Sequence Length, Nnodes (30), NFeatures(tba)]
        # self.OccMap = torch.empty(0, Nnodes,3, device=device)


    def add(self, scene, target, adj_mat):
        self.Scene = torch.cat((self.Scene, scene.unsqueeze(0)), dim=0)
        self.Target = torch.cat((self.Target, target.unsqueeze(0)), dim=0)
        self.Adj_Mat = torch.cat((self.Adj_Mat, adj_mat.unsqueeze(0)), dim=0)
        # self.OccMap = torch.cat((self.OccMap, occ_map.unsqueeze(0)), dim=0)

    def save(self):
        torch.save(self.Scene, os.path.join(self.path, 'Scene.pt'))
        torch.save(self.Target, os.path.join(self.path, 'Target.pt'))
        torch.save(self.Adj_Mat, os.path.join(self.path, 'Adj_Mat.pt'))
        # torch.save(self.OccMap, os.path.join(path, 'OccMap.pt'))
    
    def load(self):
        self.Scene = torch.load(os.path.join(self.path, 'Scene.pt')).to(self.device)
        self.Target = torch.load(os.path.join(self.path, 'Target.pt')).to(self.device)
        self.Adj_Mat = torch.load(os.path.join(self.path, 'Adj_Mat.pt')).to(self.device)
        # self.OccMap = torch.load(os.path.join(path, 'OccMap.pt'))

class Scene(Dataset):
    def __init__(self,scene, target, adj_mat): # Features include ['BBX','BBY','W', 'L' , 'Cls', 'Xreal', 'Yreal']
        self.Scene = scene
        self.Adj_Mat = adj_mat
        self.Target = target
        # self.OccMap = OccMap

    def addnoise(self, folds, amp, ratio):
        ss = self.Scene.size()
        tt = self.Target.size()
        init_scene = self.Scene.clone().detach()
        init_Adj_Mat_Scene = self.Adj_Mat
        init_Target = self.Target.clone().detach()
        # init_OccMap = self.OccMap.clone()
        mask = torch.zeros_like(init_scene)
        mask[:,:,:, :2] = 1
        mask = mask * (init_scene!= 0)
        mask_target = torch.zeros_like(init_Target)
        mask_target[:,:,:, :2] = 1
        mask_target = mask_target * (init_Target!= 0)
        for i in range(folds):
            noise = torch.rand(ss, device=self.Scene.device)*amp
            rand = torch.rand(ss, device=self.Scene.device) > ratio
            noise_target = torch.rand(tt, device=self.Scene.device)*amp
            rand_target = torch.rand(tt, device=self.Scene.device) > (ratio*0.5)
            noisy_scene = init_scene + noise * rand * mask
            noisy_target = init_Target + noise_target * rand_target * mask_target
            self.Scene = torch.cat((self.Scene, noisy_scene), dim=0)
            self.Adj_Mat = torch.cat((self.Adj_Mat, init_Adj_Mat_Scene), dim=0)
            self.Target = torch.cat((self.Target, noisy_target), dim=0)
            # self.OccMap = torch.cat((self.OccMap, init_OccMap), dim=0)

    def __len__(self):
        return self.Scene.size(0)
    
    def __getitem__(self, idx):
        return self.Scene[idx], self.Target[idx], self.Adj_Mat[idx] #, self.OccMap[idx]
    

   
def prep_data(Scene_, indices, config):
    scene = Scene_.Scene.clone()
    target = Scene_.Target.clone()
    adj_mat = Scene_.Adj_Mat.clone()
    # occ_map = Scene_clss.OccMap.clone()
    
    test_len = int(len(indices)*0.8)

    scene_tr = Scene(scene[indices[:test_len]], target[indices[:test_len]], adj_mat[indices[:test_len]])
    scene_tr.addnoise(config['noise_multiply'], config['noise_amp'], config['noise_probability'])
    scene_tst = Scene(scene[indices[test_len:]], target[indices[test_len:]], adj_mat[indices[test_len:]])
    torch.save(scene_tr, os.path.join('Pickled/Train', f'{config['sind_city']}train.pth'))
    torch.save(scene_tst, os.path.join('Pickled/Test', f'{config['sind_city']}test.pth'))
    return scene_tr, scene_tst

def read_CSV(Scene, config): # The data preprocess is a bit different from normal way on HDAAGT, the sinD author prepared it this way
    device = config['device']
    tot_len = (config['sl'] + config['future'])
    global_list = torch.zeros(config['Nusers'])
    Veh_tracks_dict, Ped_tracks_dict = read_tracks_all(config['detection_path'])
    min_frame, max_frame = 1000, 0

    for _, track in Veh_tracks_dict.items():
        frame= torch.from_numpy(track["frame_id"])
        frame_min = frame.min()
        frame_max = frame.max()
        max_frame = frame_max if frame_max > max_frame else max_frame
        min_frame = frame_min if frame_min < min_frame else min_frame
    print("Max frame",max_frame, "Min frame", min_frame)

    _, light = read_light(config['lightpath'], max_frame)
    Boundaries = torch.load(config['ZoneBoxes_path'],weights_only=True)
    Boundaries = torch.tensor(Boundaries).cpu()
    for Frme in range(min_frame,max_frame, config['sw']):
        tmp_scene = torch.zeros(tot_len, config['Nusers'],config['NFeatures'] , device = device)
        tmp_adj_mat = torch.zeros(config['sl']//config['dwn_smple'], config['Nusers'], config['Nusers'], device = device)
        global_list = {}
        order = 0
        last_Frame = Frme + tot_len
        for _, track in Veh_tracks_dict.items(): # An inefficient way to loop through the data
            id = torch.tensor(track["track_id"])
            frame= torch.from_numpy(track["frame_id"])
            x = torch.from_numpy(track["x"])
            x = (x + 66)*6 # As we treat the x, y coordinates as categorical variables, we need to scale them, this will increase the resolution of the coordinates
            y = torch.from_numpy(track["y"])
            y = (y + 82)*6
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
                for fr in range(config['sl']//config['dwn_smple']):
                    real_frame = fr+Frme
                    if real_frame in frame and id not in global_list:
                        indices = (frame >= real_frame) * (frame < last_Frame)
                        st_indx = torch.where(frame == real_frame)
                        end_indx = torch.where(frame == last_Frame)
                        if order < config['Nusers']:
                            global_list[id] = order
                            ll = light[last_Frame-sum(indices):last_Frame]
                            tmp_scene[fr: tot_len, order] = torch.stack([x[indices],y[indices],vx[indices],vy[indices],yaw_rad[indices],ax[indices],ay[indices],a_lon[indices],a_lat[indices],
                                                                    v_lon[indices],v_lat[indices],heading_rad[indices],ll[:,0], ll[:,1]], dim=1)
                            
                            order +=1
                            break
        tmp_adj_mat[:,:order, :order] = torch.eye(order, order, device=device).unsqueeze(0).repeat(config['sl']//config['dwn_smple'], 1, 1)
        Scene.add(tmp_scene[:config['sl']:config['dwn_smple']], tmp_scene[config['sl']::config['dwn_smple']], tmp_adj_mat)        
    Scene.save()
    return Scene


def loadcsv(frmpath, Header, trjpath = None):
    df = pd.read_csv(frmpath, dtype =float)
    df.columns = Header
    if trjpath is not None:
        trj = pd.read_csv(trjpath, dtype =float)
        return df, trj
    # with open(csvpath, 'r',newline='') as file:
    #     for line in file:
    #         row = line.strip().split(',')
    #         # rowf = [float(element) for element in row]
    #         # rowf = [0 if math.isnan(x) else x for x in rowf]
    #         df.append(row)
    return df


def savecsvresult(pred , groundx, groundy):
    cwd = os.getcwd()
    ct = datetime.datetime.now().strftime(r"%m%dT%H%M")
    csvpath = os.path.join(cwd,'Processed',f'Predicteddata{ct}.csv')
    with open(csvpath, 'w',newline='') as file:
        writer = csv.writer(file)
        for i in range(len(groundx)):
            rowx = pred[i,:,0].tolist()
            rowy = pred[i,:,1].tolist()
            grx = groundx[i,:,0].tolist()
            gry = groundx[i,:,1].tolist()
            grxx = groundy[i,:,0].tolist()
            gryy = groundy[i,:,1].tolist()
            writer.writerow([rowx,grxx,grx])
            writer.writerow([rowy,gryy,gry])




def savelog(log, ct): # append the log to the existing log file while keeping the old logs
    # if the log file does not exist, create one
    print(log)
    if not os.path.exists(os.path.join(os.getcwd(),'logs')):
        os.mkdir(os.path.join(os.getcwd(),'logs'))
    with open(os.path.join(os.getcwd(),'logs', f'log-{ct}.txt'), 'a') as file:
        file.write('\n' + log)
        file.close()

def Zoneconf(path = '/home/abdikhab/New_Idea_Traj_Pred/utilz/ZoneConf.yaml'):
    ZoneConf = []
    with open(path) as file:
        ZonesYML = yaml.load(file, Loader=yaml.FullLoader)
        #convert the string values to float
        for _, v in ZonesYML.items():
            lst = []
            for _, p  in v.items():    
                for x in p[0]:
                    b = re.split(r'[,()]',p[0][x])
                    lst.append((float(b[1]), float(b[2])))
            ZoneConf.append(lst)
    return ZoneConf

def zonefinder(BB, Zones):
    B, Nnodes,_ = BB.size()
    BB = BB.reshape(-1,2).cpu()
    PredZone = torch.zeros(B*Nnodes, device=BB.device)
    for n , bb in enumerate(BB.int()):
        for i, zone in enumerate(Zones):
            Poly = Polygon(zone)
            if Poly.contains(Point(bb[0], bb[1])):
                PredZone[n] = i+1
                break
        
    
    return PredZone.reshape(B, Nnodes)

def read_zones(self, path= 'utilz/ZoneConf.yaml'):
    ZoneConf = []
    with open(path) as file:
        ZonesYML = yaml.load(file, Loader=yaml.FullLoader)
        #convert the string values to float
        for _, v in ZonesYML.items():
            lst = []
            for _, p  in v.items():    
                for x in p[0]:
                    b = re.split(r'[,()]',p[0][x])
                    lst.append((float(b[1]), float(b[2])))
            ZoneConf.append(lst)
    return ZoneConf
    
def Zone_compare(Pred, Target, PrevZone, BB):
    # possiblemoves = torch.tensor([0,2,5,7,8],[0,1,2,7,8],[2],[0,2,3,5,8],[0,2,4,5,7], [5],[0,5,6,7,8],[7],[8])
    singlezones = torch.tensor([3,6,8,9], device=Pred.device)
    neighbours = torch.tensor([[0],[1],[6],[7],[8],[9],[2],[3],[4],[5]], device=Pred.device)
    B, Nnodes = Pred.size()
    Pred = Pred.reshape(-1)
    Target = Target.reshape(-1).cpu()
    PrevZone = PrevZone.reshape(-1).cpu()
    totallen = B*Nnodes
    count = 0
    nonzero = 0
    doublezone = 0
    for i in range(B*Nnodes):
        if Target[i] != 0:
            nonzero += 1
            if Pred[i] == Target[i]:
                    count += 1
            else:
                if PrevZone[i] in singlezones:
                    totallen -= 1
                    # print("Single Zone")
                if Pred[i]== neighbours[Target[i].int()]:
                    doublezone += 1
                
    return count, totallen, B*Nnodes, nonzero, doublezone

def occupancyMap(scene, Nnodes, Boundaries):
    occmap = torch.zeros(Nnodes, device=scene.device)
    for n, BB in enumerate(scene.cpu()):
        for i, bound in enumerate(Boundaries):
            Poly = Polygon(bound)
            if Poly.contains(Point(BB[0], BB[1])):
                occmap[n] = i+1
                break
    return occmap

def fast_occmap(scene, Nnodes):
    occmap = torch.zeros(Nnodes,3, device=scene.device) # Zero is reserved for no zone
    for n, BB in enumerate(scene):
        x = BB[0]
        y = BB[1]
        if x == 0 and y == 0:
            occmap[n] = torch.tensor([0, 0, 0], device=scene.device)
        else:
            if y >= -13 and y <= 6:
                if x >= -65 and x <= 23:
                    occmap[n] = torch.tensor([(int(y+13)*88 + int(x+65) + 1), (y+13)%1, (x+65)%1], device=scene.device)

            elif y >= -80 and y <= -13:
                if x >= -30 and x <= -10:
                    occmap[n]= torch.tensor([(int(y+80)*20 + int(x+30) + 1674), (y+80)%1, (x+30)%1], device=scene.device)

            elif y >= 6 and y <= 75:
                if x >= -30 and x <= -10:
                    occmap[n] = torch.tensor([(int(y-6)*20 + int(x+30) + 3015), (y-6)%1, (x+30)%1], device=scene.device)
    return occmap


def inv_occmap():
    boxes = {}
    boxes[0] = [(0, 0), (0, 0), (0, 0), (0, 0)]
    n = 1
    for y in torch.arange(-13, 6):
        for x in torch.arange(-65, 23):
            boxes[n] = [(x, y), (x+1, y), (x+1, y+1), (x, y+1)]
            n += 1
    for y in torch.arange(-80, -13):
        for x in torch.arange(-30, -10):
            boxes[n] = [(x, y), (x+1, y), (x+1, y+1), (x, y+1)]
            n += 1
    for y in torch.arange(6, 75):
        for x in torch.arange(-30, -10):
            boxes[n] = [(x, y), (x+1, y), (x+1, y+1), (x, y+1)]
            n += 1
    torch.save(boxes, os.path.join('Pickled', 'InvOccMap.pth'))
    print("len boxes", len(boxes))
    return boxes


def read_boundaries():
    boxes = []
    train_loader = torch.load(os.path.join('Pickled', 'Train', 'Changchuntrain.pth'))
    max_x = 0
    min_x = 1000
    max_y = 0
    min_y = 1000
    for sc, target , _ in train_loader:
        max_x = sc[:,:,0].max() if sc[:,:,0].max() > max_x else max_x
        max_x = target[:,:,0].max() if target[:,:,0].max() > max_x else max_x
        min_x = sc[:,:,0].min() if sc[:,:,0].min() < min_x else min_x
        min_x = target[:,:,0].min() if target[:,:,0].min() < min_x else min_x
        max_y = sc[:,:,1].max() if sc[:,:,1].max() > max_y else max_y
        max_y = target[:,:,1].max() if target[:,:,1].max() > max_y else max_y
        min_y = sc[:,:,1].min() if sc[:,:,1].min() < min_y else min_y
        min_y = target[:,:,1].min() if target[:,:,1].min() < min_y else min_y
    
    print("Max X", max_x, "Min X", min_x, "Max Y", max_y, "Min Y", min_y)
    Boundaries = [[(min_x, max_x), (min_y, max_y)]]
    torch.save (Boundaries, os.path.join('Pickled', 'Boundaries.pth'))
    for i in torch.arange(min_x, max_x, 1):
        for j in torch.arange(min_y, max_y, 1):
            boxes.append([(i, j), (i+1, j), (i+1, j+1), (i, j+1)])
    torch.save (boxes, os.path.join('Pickled', 'Boxes.pth'))
    print("len boxes", len(boxes))


def target_mask(trgt, num_head, device="cuda:3"):
    B, SL, Nnodes, _= trgt.size()
    flg = torch.ones_like(trgt[:,:,:,0]) #torch.logical_not((trgt[:,:,:,0] == 396) * (trgt[:,:,:,1] == 492)) #torch.logical_or(trgt[:,:,:,0] != 396, trgt[:,:,:,1] != 492)
    mask = 1- torch.triu(torch.ones(SL, SL, device=trgt.device), diagonal=1).unsqueeze(0).unsqueeze(0)  # Upper triangular matrix
    mask = mask.unsqueeze(4) * flg.unsqueeze(1).unsqueeze(3)
    if num_head > 1:
        mask = mask.repeat_interleave(num_head, dim=0)
    return mask == 0

def create_src_mask(src, device="cuda:3"):
    mask = torch.logical_or(src[:,:,:, 0] == 0, src[:,:,:, 1] == 0)
    return mask


def read_tracks_all(path):

    Veh_path = os.path.join(path, 'Veh_smoothed_tracks.csv')
    Ped_path = os.path.join(path, 'Ped_smoothed_tracks.csv')
    veh_df = pd.read_csv(Veh_path)
    veh_state_name = veh_df.columns.tolist()
    assert veh_state_name == ['track_id', 'frame_id', 'timestamp_ms', 'agent_type', 'x', 'y', 'vx', 'vy', 'yaw_rad',
                         'heading_rad', 'length', 'width', 'ax', 'ay', 'v_lon', 'v_lat', 'a_lon', 'a_lat']

    Veh_tracks_dict = {}
    Veh_tracks_group = veh_df.groupby(["track_id"], sort=False)
    for track_id, group in Veh_tracks_group:
        track = {}
        track_dict = group.to_dict(orient="list")
        for key, value in track_dict.items():
            if key in ["track_id", "agent_type"]:

                track[key] = value[0]

            else:
                track[key] = np.array(value)

        track["center"] = np.stack([track["x"], track["y"]], axis=-1)
        track["bbox"], track["triangle"] = calculate_rot_bboxes_and_triangle(track["x"], track["y"],
                                                                    track["length"], track["width"],
                                                                    track["yaw_rad"])
        Veh_tracks_dict[track_id] = track #track["yaw_rad"]

    ped_df = pd.read_csv(Ped_path)
    statename = ped_df.columns.tolist()
    assert statename == ['track_id', 'frame_id', 'timestamp_ms', 'agent_type', 'x', 'y', 'vx', 'vy', 'ax', 'ay']
    Ped_tracks_dict = {}
    Ped_tracks_group = ped_df.groupby(["track_id"], sort=False)
    for track_id, group in Ped_tracks_group:
        track = {}
        track_dict = group.to_dict(orient="list")
        for key, value in track_dict.items():
            if key in ["track_id", "agent_type"]:
                track[key] = value[0]
            else:
                track[key] = np.array(value)
        track["center"] = np.stack([track["x"], track["y"]], axis=-1)
        track["bbox"], _ = calculate_rot_bboxes_and_triangle(track["x"], track["y"])

        Ped_tracks_dict[track_id] = track

    return Veh_tracks_dict, Ped_tracks_dict



def read_light(path, maxframe):

    df_light = pd.read_csv(path)
    light_tensor = torch.zeros(maxframe+101, 2)
    light_dict = {}
    memory = (0,0)
    frame = 0
    flag = 0

    for row in df_light.itertuples():
        if row[1] < frame:
            memory = row[3:]
            continue
        while frame < row[1]:
            light_dict[frame] = memory
            light_tensor[frame] = torch.tensor(memory)
            frame += 1
            if frame > maxframe + 100:
                flag = 1
                break
        memory = row[3:]
        if flag == 1:
            break

    return light_dict, light_tensor


def read_tracks_meta(path):

    tracks_meta_df = pd.read_csv(path, index_col="trackId")

    tracks_meta_dict = tracks_meta_df.to_dict("index")

    return tracks_meta_dict



def calculate_rot_bboxes_and_triangle(center_points_x, center_points_y, length=0.5, width=0.5, rotation=0):
    """
    Calculate bounding box vertices and triangle vertices from centroid, width and length.

    :param centroid: center point of bbox
    :param length: length of bbox
    :param width: width of bbox
    :param rotation: rotation of main bbox axis (along length)
    :return:
    """

    centroid = np.array([center_points_x, center_points_y]).transpose()#(n, 2)

    centroid = np.array(centroid)
    if centroid.shape == (2,):
        centroid = np.array([centroid])

    # Preallocate
    data_length = centroid.shape[0]
    rotated_bbox_vertices = np.empty((data_length, 4, 2))

    # Calculate rotated bounding box vertices
    rotated_bbox_vertices[:, 0, 0] = -length / 2
    rotated_bbox_vertices[:, 0, 1] = -width / 2

    rotated_bbox_vertices[:, 1, 0] = length / 2
    rotated_bbox_vertices[:, 1, 1] = -width / 2

    rotated_bbox_vertices[:, 2, 0] = length / 2
    rotated_bbox_vertices[:, 2, 1] = width / 2

    rotated_bbox_vertices[:, 3, 0] = -length / 2
    rotated_bbox_vertices[:, 3, 1] = width / 2

    for i in range(4):
        th, r = cart2pol(rotated_bbox_vertices[:, i, :])
        rotated_bbox_vertices[:, i, :] = pol2cart(th + rotation, r).squeeze()
        rotated_bbox_vertices[:, i, :] = rotated_bbox_vertices[:, i, :] + centroid #(n, 4, 2)

    # Calculate triangle vertices
    triangle_factor = 0.75

    a = rotated_bbox_vertices[:, 3, :] + (
                (rotated_bbox_vertices[:, 2, :] - rotated_bbox_vertices[:, 3, :]) * triangle_factor)
    b = rotated_bbox_vertices[:, 0, :] + (
                (rotated_bbox_vertices[:, 1, :] - rotated_bbox_vertices[:, 0, :]) * triangle_factor)
    c = rotated_bbox_vertices[:, 2, :] + ((rotated_bbox_vertices[:, 1, :] - rotated_bbox_vertices[:, 2, :]) * 0.5)

    triangle_array = np.array([a, b, c]).swapaxes(0, 1) #(3, n, 2)


    return rotated_bbox_vertices, triangle_array


def cart2pol(cart):
    """
    Transform cartesian to polar coordinates.
    :param cart: Nx2 ndarray
    :return: 2 Nx1 ndarrays
    """
    if cart.shape == (2,):
        cart = np.array([cart])

    x = cart[:, 0]
    y = cart[:, 1]

    th = np.arctan2(y, x)
    r = np.sqrt(np.power(x, 2) + np.power(y, 2))
    return th, r


def pol2cart(th, r):
    """
    Transform polar to cartesian coordinates.
    :param th: Nx1 ndarray
    :param r: Nx1 ndarray
    :return: Nx2 ndarray
    """

    x = np.multiply(r, np.cos(th))
    y = np.multiply(r, np.sin(th))

    cart = np.array([x, y]).transpose()
    return cart


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def Find_topk_selected_words (Pred_target, Target): # This function is used to find the top k selected words from the predicted target, only for sinD dataset as we have *6 resolution increase
    Word_Probs = Pred_target.softmax(dim=-1)
    top_values, top_indices = torch.topk(Word_Probs, k = 5, dim=-1)
    Topk_Selected_words = (top_indices*top_values).sum(-1)/top_values.sum(-1)
    flg = Target[:,:,:,:1] != 0 # We have blank rows in the data as the number of present agents changes during time
    ADE = torch.sqrt(torch.pow((Topk_Selected_words*flg -Target),2).sum(-1)).mean()/6
    FDE = torch.sqrt(torch.pow((Topk_Selected_words*flg -Target),2).sum(-1)[:,-1]).mean()/6 # This 6 is the resolution increase that we did in the data
    return Topk_Selected_words, ADE, FDE

if __name__ == "__main__":
    print("Yohoooo, Ran a Wrong Script!")