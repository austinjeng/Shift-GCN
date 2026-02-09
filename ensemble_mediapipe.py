import argparse
import pickle

import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

label = open('./data/mediapipe/val_label.pkl', 'rb')
label = np.array(pickle.load(label))
r1 = pickle.load(open('./work_dir/mediapipe_ShiftGCN_joint/eval_results/best_acc.pkl', 'rb'))
r2 = pickle.load(open('./work_dir/mediapipe_ShiftGCN_bone/eval_results/best_acc.pkl', 'rb'))
r3 = pickle.load(open('./work_dir/mediapipe_ShiftGCN_joint_motion/eval_results/best_acc.pkl', 'rb'))
r4 = pickle.load(open('./work_dir/mediapipe_ShiftGCN_bone_motion/eval_results/best_acc.pkl', 'rb'))

alpha = [0.6, 0.6, 0.4, 0.4]

right_num = total_num = right_num_5 = 0
all_preds = []
all_labels = []
for i in tqdm(range(len(label[0]))):
    name = label[0][i]
    l = label[1][i]
    r11 = r1[name]
    r22 = r2[name]
    r33 = r3[name]
    r44 = r4[name]
    r = r11 * alpha[0] + r22 * alpha[1] + r33 * alpha[2] + r44 * alpha[3]
    rank_5 = r.argsort()[-5:]
    right_num_5 += int(int(l) in rank_5)
    r = np.argmax(r)
    right_num += int(r == int(l))
    total_num += 1
    all_preds.append(r)
    all_labels.append(int(l))

acc = right_num / total_num
acc5 = right_num_5 / total_num
print('top1: ', acc)
print('top5: ', acc5)

# Binary classification metrics
print('\n--- Classification Report ---')
print(classification_report(all_labels, all_preds,
                            target_names=['Non-Fall', 'Fall'], digits=4))

cm = confusion_matrix(all_labels, all_preds)
print('--- Confusion Matrix ---')
print(f'              Pred Non-Fall  Pred Fall')
print(f'  Non-Fall    {cm[0,0]:>12}  {cm[0,1]:>9}')
print(f'  Fall        {cm[1,0]:>12}  {cm[1,1]:>9}')
