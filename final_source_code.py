####################################################################################
# 웹에서 실제 입력된 영상에서 여러 전처리 과정 이후 DB와 유사도 검색을 진행(최종버전)
# Classification으로 해결한 코드 V.2.0.0
####################################################################################

import cv2
import time
import argparse
import os
import torch
import posenet
import json
import math
import numpy as np
import torch.nn as nn

from netvlad import NetVLAD
from netvlad import EmbedNet

from torch.autograd import Variable
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--notxt', action='store_true')
parser.add_argument('--video_dir', type=str, default='./uploads')
parser.add_argument('--image_dir', type=str, default='./views/test_in')
parser.add_argument('--output_dir', type=str, default='./views/test_out')
parser.add_argument('--compare_id', type=int, default=-1)
args = parser.parse_args()

####################################################################################
# BLSTM으로 추출된 feature의 metric learning을 적용함
# NetVlad : https://github.com/lyakaap/NetVLAD-pytorch
# 언어 : 파이토치
####################################################################################

def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: Variable of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: Variable of shape (batch_size, batch_size)
    """

    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = torch.mm(embeddings, embeddings.t())

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = torch.diag(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = torch.unsqueeze(square_norm, 0) - 2.0 * dot_product + torch.unsqueeze(square_norm, 1)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = torch.clamp(distances, min=0.0)

    if not squared:
        # Not sure if needed for pytorch but does not harm
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = torch.eq(distances, 0.0).float()
        distances = distances + mask * 1e-16

        distances = torch.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances

def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Args:
        labels: torch.Tensor with shape [batch_size]
    Returns:
        mask: Varieble with torch.ByteTensor with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = torch.eye(labels.size(0)).bool()
    if labels.is_cuda:
        indices_equal = indices_equal.cuda()
    indices_not_equal = ~indices_equal

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))

    # Combine the two masks
    mask = indices_not_equal & labels_equal

    return mask


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: torch.Tensor with shape [batch_size]
    Returns:
        mask: Variable with torch.ByteTensor with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))

    mask = ~labels_equal

    return mask


def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: torch.Tensor with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = torch.eye(labels.size(0)).byte()
    if labels.is_cuda:
        indices_equal = indices_equal.cuda()
    indices_not_equal = ~indices_equal
    i_not_equal_j = torch.unsqueeze(indices_not_equal, 2)
    i_not_equal_k = torch.unsqueeze(indices_not_equal, 1)
    j_not_equal_k = torch.unsqueeze(indices_not_equal, 0)

    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k


    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
    #if labels.is_cuda:
    #    label_equal = label_equal.cuda()
    i_equal_j = torch.unsqueeze(label_equal, 2)
    i_equal_k = torch.unsqueeze(label_equal, 1)

    valid_labels = i_equal_j & (~i_equal_k)
    
    # Combine the two masks
    mask = distinct_indices & valid_labels.type(torch.uint8)

    return mask

def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.
    We generate all the valid triplets and average the loss over the positive ones.
    Args:
        labels: Variable with labels of the batch, of size (batch_size,)
        embeddings: Variable with tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    anchor_positive_dist = torch.unsqueeze(pairwise_dist, 2)
    anchor_negative_dist = torch.unsqueeze(pairwise_dist, 1)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels)
    mask = Variable(mask.float())
    triplet_loss = mask * triplet_loss

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = torch.clamp(triplet_loss, min=0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = torch.gt(triplet_loss, 1e-16)
    num_positive_triplets = valid_triplets.sum().float()
    num_valid_triplets = mask.sum()
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets

def batch_hard_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.
    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    Args:
        labels: Variable with labels of the batch, of size (batch_size,)
        embeddings: Variable with tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = Variable(mask_anchor_positive.float())

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = mask_anchor_positive * pairwise_dist

    # shape (batch_size, 1)
    hardest_positive_dist, _ = torch.max(anchor_positive_dist, dim=1, keepdim=True)

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = Variable(mask_anchor_negative.float())

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist, _ = torch.max(pairwise_dist, dim=1, keepdim=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist, _ = torch.min(anchor_negative_dist, dim=1, keepdim=True)

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = torch.clamp(hardest_positive_dist - hardest_negative_dist + margin, min=0.0)

    # Get final mean triplet loss
    triplet_loss = triplet_loss.mean()

    return triplet_loss

####################################################################################
# 시퀀스 Hand Craft Feature 데이터에서 새로운 feature를 추출하는 딥러닝 네트워크
# BLSTM ( bbidirectional= True)
# 언어 : 파이토치
####################################################################################

class LSTM(nn.Module):
 
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
                    num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
 
        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=True)
 
        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim*2, output_dim)
 
    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers*2, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers*2, self.batch_size, self.hidden_dim))
 
    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        input = input.float()
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
        feature_map = (lstm_out[-1].view(self.batch_size, -1))
        feature_map = feature_map.view(feature_map.size(0),feature_map.size(1),1,1)
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))

        return y_pred,feature_map


####################################################################################
# Hand Craft Feature 계산 수식
####################################################################################

def distance(x1, y1, x2, y2):
    result = math.sqrt( math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
    return result

def dotproduct(x1, y1, x2, y2):
    result = x1*x2 + y1*y2
    return result

def crossproduct(ax, ay, bx, by, cx, cy):
    a = ax*by+bx*cy+cx*ay
    b = ay*bx+by*cx+cy*ax
    return a-b

def arccos(n):
    result = math.acos(n)
    return result

def arcsin(n):
    result = math.asin(n)
    return result

def absolutevalue(x1, y1, x2, y2):
    v1 = math.sqrt(math.pow(x1, 2) + math.pow(y1, 2))
    v2 = math.sqrt(math.pow(x2, 2) + math.pow(y2, 2))
    result = v1*v2
    return result

def xy_to_feature_1(leftshoulder, rightshoulder, lefthip, righthip):
    
    d_shoulder = distance(leftshoulder[0], leftshoulder[1], rightshoulder[0], rightshoulder[1])
    d_hip = distance(lefthip[0], lefthip[1], righthip[0], righthip[1])

    return d_shoulder/d_hip

def xy_to_feature_2(leftshoulder, rightshoulder, leftelbow, rightelbow) :
    Al = (leftelbow[0]-leftshoulder[0], leftelbow[1]-leftshoulder[1])
    Ar = (rightelbow[0]-rightshoulder[0], rightelbow[1]-rightshoulder[1])

    return ((2*math.pi)-arccos(dotproduct(Al[0], Al[1], Ar[0], Ar[1])/absolutevalue(Al[0], Al[1], Ar[0], Ar[1])))/(2*math.pi)

def xy_to_feature_3(lefthip, righthip, leftknee, rightknee) :
    
    Ll = (leftknee[0]-lefthip[0], leftknee[1]-lefthip[1])
    Lr = (rightknee[0]-righthip[0], rightknee[1]-righthip[1])
    
    return (arccos(dotproduct(Ll[0], Ll[1], Lr[0], Lr[1])/absolutevalue(Ll[0], Ll[1], Lr[0], Lr[1])))/(math.pi)

def xy_to_feature_4(lefthip, righthip, leftshoulder, rightshoulder) :
    Pcenterhip = ((lefthip[0]+righthip[0])/2, (lefthip[1]+righthip[1])/2)
    neck = ((leftshoulder[0]+rightshoulder[0])/2, (leftshoulder[1]+rightshoulder[1])/2)
    h = (neck[0]-Pcenterhip[0], neck[1]-Pcenterhip[1])
    x = (1,0)
    return (arccos(dotproduct(h[0], h[1], x[0], x[1])/absolutevalue(h[0], h[1], x[0], x[1])))/(math.pi)

def xy_to_feature_5(leftshoulder, rightshoulder, leftelbow, rightelbow, leftwrist, rightwrist) :
    Al = (leftshoulder[0]-leftelbow[0], leftshoulder[1]-leftelbow[1])
    Ar = (rightshoulder[0]-rightelbow[0], rightshoulder[1]-rightelbow[1])
    Wl = (leftwrist[0]-leftelbow[0], leftwrist[1]-leftelbow[1])
    Wr = (rightwrist[0]-rightelbow[0], rightwrist[1]-rightelbow[1])
    leftelbowangle = (arcsin(crossproduct(leftelbow[0],leftelbow[1],leftshoulder[0],leftshoulder[1],leftwrist[0],leftwrist[1])/absolutevalue(Al[0], Al[1], Wl[0], Wl[1])))/(math.pi)
    rightelbowangle = (arcsin(crossproduct(rightelbow[0],rightelbow[1],rightshoulder[0],rightshoulder[1],rightwrist[0],rightwrist[1])/absolutevalue(Ar[0], Ar[1], Wr[0], Wr[1])))/(math.pi)

    return [leftelbowangle, rightelbowangle]

def xy_to_feature_6(lefthip, righthip, leftknee, rightknee, leftankle, rightankle):

    Ll = (lefthip[0]-leftknee[0], lefthip[1]-leftknee[1])
    Lr = (righthip[0]-rightknee[0], righthip[1]-rightknee[1])

    Cl = (leftankle[0]-leftknee[0], leftankle[1]-leftknee[1])
    Cr = (rightankle[0]-rightknee[0], rightankle[1]-rightknee[1])

    leftkneeangle = (arcsin(crossproduct(leftknee[0],leftknee[1],lefthip[0],lefthip[1],leftankle[0],leftankle[1])/absolutevalue(Ll[0], Ll[1], Cl[0], Cl[1])))/(math.pi)
    rightkneeangle = (arcsin(crossproduct(rightknee[0],rightknee[1],righthip[0],righthip[1],rightankle[0],rightankle[1])/absolutevalue(Lr[0], Lr[1], Cr[0], Cr[1])))/(math.pi)

    return [leftkneeangle, rightkneeangle]

####################################################################################
# 비디오를 이미지로 변환하는 함수
####################################################################################

def video2frame(invideofilename, save_path):
    vidcap = cv2.VideoCapture(invideofilename)
    count = 0
    while True:
      success,image = vidcap.read()
      if not success:
          break
      # print ('Read a new frame: ', success)
      fname = "{}.jpg".format("{0:05d}".format(count))
      cv2.imwrite(save_path + fname, image) # save frame as JPEG file
      count += 1
    # print("{} images are extracted in {}.". format(count, save_path))

def findIndex(final_list, index):
    flag = 1

    for i in range(len(final_list)):
        if (final_list[i] == index):
            flag = 0

    if(flag == 1):
        return 1
    
    else :
        return 0
    
    

def main():
    
    test_total_class = list()

    # 미리 학습한 모델을 불러와 스켈레톤 벡터 추출 준비(PoseNet)

    posenet_model = posenet.load_model(args.model)
    posenet_model = posenet_model.cuda()
    output_stride = posenet_model.output_stride
    
    # 비디오를 이미지로 변환

    video_filenames = [v.path for v in os.scandir(args.video_dir) if v.is_file() and v.path.endswith(('.mp4'))]
   
    if args.image_dir:
        if not os.path.exists(args.image_dir):
            os.makedirs(args.image_dir)

    for iv,v in enumerate(video_filenames):
        if not os.path.exists(args.image_dir+'/'+v[10:-4]+'/'):
            os.makedirs(args.image_dir+'/'+v[10:-4]+'/')
        video2frame(v,args.image_dir+'/'+v[10:-4]+'/')

        if args.output_dir:
          if not os.path.exists(args.output_dir+'/'+v[10:-4]+'/'):
            os.makedirs(args.output_dir+'/'+v[10:-4]+'/')

    # 비디오에서 추출된 이미지를 통한 스켈레톤 벡터 추출 시작

    for iv,v in enumerate(video_filenames):
      filenames = [f.path for f in os.scandir(args.image_dir+'/'+v[10:-4]+'/') if f.is_file() and f.path.endswith(('.png', '.jpg'))]
      for i,f in enumerate(filenames): 
          input_image, draw_image, output_scale = posenet.read_imgfile(
              f, scale_factor=args.scale_factor, output_stride=output_stride)

          with torch.no_grad():
              input_image = torch.Tensor(input_image).cuda()
                # PoseNet을 통해 벡터 추출
              heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = posenet_model(input_image)

              pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                  heatmaps_result.squeeze(0),
                  offsets_result.squeeze(0),
                  displacement_fwd_result.squeeze(0),
                  displacement_bwd_result.squeeze(0),
                  output_stride=output_stride,
                  max_pose_detections=10,
                  min_pose_score=0.25)

          keypoint_coords *= output_scale
            # 스켈레톤 벡터 추출 시각화를 위한 이미지저장(이미지-추출한 시퀀스 이미지에서 댄서의 스켈레톤 벡터를 표시)
          if args.output_dir:
              draw_image = posenet.draw_skel_and_kp(draw_image, pose_scores, keypoint_scores, keypoint_coords,min_pose_score=0.25, min_part_score=0.25)
              cv2.imwrite(os.path.join(args.output_dir, os.path.relpath(f, args.image_dir)), draw_image)

          if not args.notxt:
              max_score = 0
              max_index = 0
              ignore = 0

              for pi in range(len(pose_scores)):
                  if max_score > pose_scores[pi] :
                      max_index = pi

                  if pose_scores[pi] == 0.:
                      ignore = 1
                      break
              
              # Posenet을 통해 추출한 스켈레톤 벡터를 가지고 Hand Craft Feature 계산
              if pose_scores[max_index] != 0. :
                  tmp_data = dict()
                  out_data = dict(image_name=[f[10:-4]])

                  for ki, (s, c) in enumerate(zip(keypoint_scores[max_index, :], keypoint_coords[max_index, :, :])):
                      tmp_data[posenet.PART_NAMES[ki]] = c.tolist()

                  out_data['feature_1'] = xy_to_feature_1(tmp_data['leftShoulder'], tmp_data['rightShoulder'], tmp_data['leftHip'], tmp_data['rightHip'])
                  out_data['feature_2'] = xy_to_feature_2(tmp_data['leftShoulder'], tmp_data['rightShoulder'], tmp_data['leftElbow'], tmp_data['rightElbow'])
                  out_data['feature_3'] = xy_to_feature_3(tmp_data['leftHip'], tmp_data['rightHip'], tmp_data['leftKnee'], tmp_data['rightKnee'])
                  out_data['feature_4'] = xy_to_feature_4(tmp_data['leftHip'], tmp_data['rightHip'], tmp_data['leftShoulder'], tmp_data['rightShoulder'])
                  out_data['feature_5'] = xy_to_feature_5(tmp_data['leftShoulder'], tmp_data['rightShoulder'], tmp_data['leftElbow'], tmp_data['rightElbow'], tmp_data['leftWrist'], tmp_data['rightWrist'])
                  out_data['feature_6'] = xy_to_feature_6(tmp_data['leftHip'], tmp_data['rightHip'], tmp_data['leftKnee'], tmp_data['rightKnee'], tmp_data['leftAnkle'], tmp_data['rightAnkle'])
                  
                  out_data['total_feature'] = list()
                  out_data['total_feature'].extend([out_data['feature_1']])
                  out_data['total_feature'].extend([out_data['feature_2']])
                  out_data['total_feature'].extend([out_data['feature_3']])
                  out_data['total_feature'].extend([out_data['feature_4']])
                  out_data['total_feature'].extend([out_data['feature_5'][0]])
                  out_data['total_feature'].extend([out_data['feature_5'][1]])
                  out_data['total_feature'].extend([out_data['feature_6'][0]])
                  out_data['total_feature'].extend([out_data['feature_6'][1]])

                  test_total_class.append(out_data['total_feature'])

                  if len(test_total_class) is 150 :
                      break
    
    
    ############################################################################
    # 유사도 검색을 진행
    ############################################################################

    # 사전에 학습된 NetVLAD 모델 로드 (데이터셋에 맞춰 코랩으로 학습진행)
    base_feature = np.load('./models/Train_for_pca_DIY.npy')
    base_feature_label = np.load('./models/Train_label_for_pca_DIY.npy')

    class_cnt = 20
    test_total_class = np.array(test_total_class)
    test_total_class = test_total_class.reshape(150,1,8)
    x_test_data = torch.from_numpy(test_total_class)

    # Pretrained BLSTM과 NetVLAD를 결합해 새로운 모델 정의(EmbedNet)
    back_bone_model = LSTM(8, 32, batch_size=1, output_dim=class_cnt, num_layers=2)
    net_vlad = NetVLAD(num_clusters=40, dim=64, alpha=1.0)
    vlad_model = EmbedNet(back_bone_model, net_vlad)
    vlad_model.load_state_dict(torch.load('./models/checkpoints/VLAD_Checkpoint_20200609_Best_DIY_total.pth'), strict=True)

    # 웹을 통해 입력된 영상에서 스켈레톤추출-Hand craft Feature계산 과정을 거치고 해당 data를 BLSTM과 NetVLAD에 입력 
    test_out_feature = vlad_model(x_test_data)
    test_out_feature = np.array(test_out_feature)
    test_out_feature = test_out_feature.reshape(-1 , 2560)

    #실제 결과는 2560차원에 데이터가 생성. 좀더 빠른 데모를 위해서 실제 2차원으로 축소 진행
    pca = PCA(n_components=2)
    X_Train_pca = pca.fit_transform(base_feature)
    X_test_pca = pca.transform(test_out_feature)

    check = np.concatenate((X_Train_pca,X_test_pca),axis=0)

    #축소된 데이터들과 기존 DB간의 similarity 계산
    pairwise_dist_t = _pairwise_distances(torch.from_numpy(check))
    pairwise_dist_t = pairwise_dist_t.cpu().detach().numpy()
    pairwise_dist_sort = np.sort(pairwise_dist_t[-1][:-1])
    

    # 계산된 distance를 상대적인 유사도 값으로 변환 및 출력
    # 특정 안무 검색과 전체 안무 검색에 따라 두가지로 나뉨
    
    final_out_bef = list()
    final_out = list()
    final_score = list()

    if(args.compare_id == -1):
        for index in range(0, 20):
    
            for i in range(len(pairwise_dist_t)) :
                if pairwise_dist_sort[index] == pairwise_dist_t[-1][i] :
                    
                    score = 100-(300*pairwise_dist_sort[index])

                    if score > 0 and findIndex(final_out, base_feature_label[i]):
                        
                        final_out_bef.append(i)
                        final_score.append(score)
                        final_out.append(base_feature_label[i])
                        
                    break
    
    print(final_out)
    print(final_score)





if __name__ == "__main__":
    main()


PART_NAMES = [
    "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
    "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
    "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
]
