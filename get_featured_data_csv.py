import pandas as pd
import numpy as np

# 손 사이즈 구하기
def get_hand_size(coords):
    coords = np.array(coords)
    x_min, y_min = np.min(coords[:, 0]), np.min(coords[:, 1])
    x_max, y_max = np.max(coords[:, 0]), np.max(coords[:, 1])
    return np.linalg.norm([x_max - x_min, y_max - y_min])

# 손가락 마디별 각도 계산
def angle_finger_joint(wrist, p1, p2, p3, p4):
    angle1 = angle_between(wrist, p1, p2)
    angle2 = angle_between(p1, p2, p3)
    angle3 = angle_between(p2, p3, p4)

    return [angle1, angle2, angle3]

# 마디간 각도를 0 ~ 360까지의 거리를 라디안으로 계산
def angle_between(p1, p2, p3):
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)

    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)

    dot = np.dot(a_norm, b_norm)
    cross = a_norm[0] * b_norm[1] - a_norm[1] * b_norm[0]  # 2D 외적: z성분

    angle = np.arctan2(cross, dot)
    if angle < 0:
        angle += 2 * np.pi

    return angle

# 손 사이즈 대비 손가락 관절 간 거리 계산
def euclidean_distance(p1, p2, hand_size):
    distance = np.linalg.norm(np.array(p1) - np.array(p2))
    result_distance = distance / hand_size
    if hand_size == 0:
        result_distance = 0.0
    return result_distance

def check_facing_camera(distance):
    if distance < 0.1:
        return distance / 2

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

    # 손가락 굽힘 각도 (손목 + 손끝까지 향하는 각 관절 4개, 0 ~ 14열)
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
    
    # 4-0-8번 관절 각도
    features.append(angle_between(coords[4], coords[0], coords[8]))
    
    # 손가락 간 거리 (15 ~ 18열)
    distances = []
    hand_size = get_hand_size(coords)
    tip_indices = [4, 8, 12, 16, 20]
    for i in range(len(tip_indices)-1):
        d = euclidean_distance(coords[tip_indices[i]], coords[tip_indices[i+1]], hand_size)
        distances.append(d)
        features.append(d)

    # 거리 비율 (19열)
    if distances[1] != 0:
        features.append(distances[0] / distances[1])
    else:
        features.append(0.0)

    # 손 회전 각도 (20열)
    features.append(hand_orientation_angle(coords))

    return pd.Series(features)

# CSV 로딩
df = pd.read_csv("digits.csv", encoding="cp949")
feature_df = df.drop(columns=['label']).apply(extract_features, axis=1)
feature_df['label'] = df['label']

# 저장
feature_df.to_csv("digit_feature.csv", index=False, encoding="cp949")
print("[✓] feature-data 저장 완료.")
