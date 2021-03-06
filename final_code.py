####################################################################################
# 웹에서 실제 입력된 영상에서 여러 전처리 과정 이후 DB와 안무검색을 진행
# Classification으로 해결한 코드 V.1.0.0
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


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--notxt', action='store_true')
parser.add_argument('--video_dir', type=str, default='./uploads')
parser.add_argument('--image_dir', type=str, default='./test_in')
parser.add_argument('--output_dir', type=str, default='./test_out')
args = parser.parse_args()

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

      fname = "{}.jpg".format("{0:05d}".format(count))
      cv2.imwrite(save_path + fname, image) # save frame as JPEG file
      count += 1


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
    # 해당 코드는 첫번째 버전으로 Classification으로 춤을 검색함
    ############################################################################

    class_cnt = 20
    test_total_class = np.array(test_total_class)
    test_total_class = test_total_class.reshape(150,1,8)            
    x_test_data = torch.from_numpy(test_total_class[:,:class_cnt,:])
    # 스켈레톤 벡터를 BLSTM을 통해 새로운 피처로 기술(이때도 사전에 학습한 pretrained weight를 사용한다.)
    base_BLSTM_model = LSTM(8, 32, batch_size=1, output_dim=class_cnt, num_layers=2)
    base_BLSTM_model.load_state_dict(torch.load('./models/checkpoints/chekcpoint_20200523_best_top3_total.pth'), strict=True)
    y_test_pred, _ = base_BLSTM_model(x_test_data)
    # topk에 결과 출력(Classificication)
    _,test_label = torch.topk(y_test_pred,3)
    
    print(test_label)




if __name__ == "__main__":
    main()


PART_NAMES = [
    "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
    "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
    "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
]
