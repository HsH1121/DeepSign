import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 자모 리스트 정의
jaum_list = list("ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ")
moum_list = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅛㅜㅠㅡㅣ")
all_jamo = sorted(set(jaum_list + moum_list + jaum_list))
char_to_index = {char: idx for idx, char in enumerate(all_jamo)}
index_to_char = {idx: char for char, idx in char_to_index.items()}

VOCAB_SIZE = len(char_to_index)
max_len = 20  # 학습 시 사용한 시퀀스 길이와 동일

# 모델 불러오기
model = load_model("SiotYu_distinction_2.h5")

# ✅ 연속 예측 함수
def predict_jamo_sequence(seed_jamo, n_predict=5):
    result = list(seed_jamo)  # 초기 입력 복사
    for _ in range(n_predict):
        input_idx = [char_to_index[j] for j in result]
        input_pad = pad_sequences([input_idx], maxlen=max_len, padding='pre')
        pred = model.predict(input_pad, verbose=0)
        pred_idx = np.argmax(pred)
        pred_jamo = index_to_char[pred_idx]
        result.append(pred_jamo)
    return result

# ✅ 사용 예시
seed_input = ['ㅇ', 'ㅏ', 'ㄹ', 'ㄱ', 'ㅔ']
predicted_sequence = predict_jamo_sequence(seed_input, n_predict=5) # n_predict : 다음 예측할 문자 개수

print("입력 자모:", seed_input)
print("예측된 전체 자모 시퀀스:", predicted_sequence)
