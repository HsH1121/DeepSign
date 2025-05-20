import pandas as pd
import numpy as np

def angle_between(p1, p2, p3):
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def hand_orientation_angle(coords):
    wrist = np.array(coords[0])
    middle_mcp = np.array(coords[9])
    vec = middle_mcp - wrist
    angle = np.arctan2(vec[1], vec[0])  # 라디안
    return angle  # 그대로 추가해도 되고 np.degrees(angle)로 바꿔도 됨

def extract_features(row):
    coords = []
    for i in range(21):
        coords.append([row[f'x{i}'], row[f'y{i}']])

    features = []

    # 손가락 굽힘 각도
    finger_joints = [(5,6,7), (9,10,11), (13,14,15), (17,18,19), (2,3,4)]
    for a, b, c in finger_joints:
        features.append(angle_between(coords[a], coords[b], coords[c]))

    # 손가락 간 각도
    angle_pairs = [(4,8), (8,12), (12,16), (16,20)]
    for i, j in angle_pairs:
        features.append(angle_between(coords[0], coords[i], coords[j]))

    # 손가락 간 거리
    distances = []
    tip_indices = [4, 8, 12, 16, 20]
    for i in range(len(tip_indices)-1):
        d = euclidean_distance(coords[tip_indices[i]], coords[tip_indices[i+1]])
        distances.append(d)
        features.append(d)

    # 거리 비율
    if distances[1] != 0:
        features.append(distances[0] / distances[1])
    else:
        features.append(0.0)

    # 손 회전 각도
    features.append(hand_orientation_angle(coords))

    return pd.Series(features)

# CSV 로딩
df = pd.read_csv("../hand_landmarks.csv")
feature_df = df.drop(columns=['label']).apply(extract_features, axis=1)
feature_df['label'] = df['label']

# 저장
feature_df.to_csv("hand_features_with_orientation.csv", index=False)
print("[✓] 손 회전 각도 포함된 feature-data 저장 완료.")
