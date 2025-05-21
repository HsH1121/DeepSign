import pandas as pd
import numpy as np

# 손가락 마디별 각도 계산
def angle_finger_joint(wrist, p1, p2, p3, p4):
    '''
    손가락 하나, 관절 4개의 좌표를 받아서
    각도1 : wrist - p1 - p2
    각도2 : p1 – p2 – p3
    각도3 : p2 – p3 – p4
    를 계산하여 리스트로 반환

    단, 각도가 angle_threshold_deg 이상(굽힘 심함)일 경우 → 90.0 반환
    '''

    angle1 = angle_between(wrist, p1, p2)
    angle2 = angle_between(p1, p2, p3)
    angle3 = angle_between(p2, p3, p4)

    return [angle1, angle2, angle3]

def angle_between(p1, p2, p3):
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))

# 손가락 굽힘 여부 확인
def check_angle(a, b, c):
    #dist_threshold = 0.02
    angle_threshold_deg = 115

    angle = angle_between(a, b, c)
    dist = np.linalg.norm(np.array(b) - np.array(c))
    if angle < np.radians(angle_threshold_deg):
        return 90.0 # 굽힘 너무 심하면(115도 미만) 접힘 처리(90 반환)
    return angle

# 손가락 관절별 거리 계산
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

    # 손가락 굽힘 각도 (손목 + 손끝까지 향하는 각 관절 4개)
    finger_joints = [
        (0, 1, 2, 3, 4),  # 엄지
        (0, 5, 6, 7, 8),  # 검지
        (0, 9, 10, 11, 12),  # 중지
        (0, 13, 14, 15, 16),  # 약지
        (0, 17, 18, 19, 20)  # 소지
    ]
    for wrist, p1, p2, p3, p4 in finger_joints:
        angles = angle_finger_joint(coords[wrist], coords[p1], coords[p2], coords[p3], coords[p4])
        features.extend(angles)

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
df = pd.read_csv("../hand_landmarks.csv", encoding="cp949")
feature_df = df.drop(columns=['label']).apply(extract_features, axis=1)
feature_df['label'] = df['label']

# 저장
feature_df.to_csv("new_hand_features_with_orientation_with_han.csv", index=False, encoding="cp949")
print("[✓] 손 회전 각도 포함된 feature-data 저장 완료.")
