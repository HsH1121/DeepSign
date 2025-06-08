import hgtk
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from sklearn.preprocessing import LabelEncoder

# 전체 한글 자모 목록
jaum_list = list("ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ")
moum_list = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅛㅜㅠㅡㅣ")

# 겹자음 분해용 매핑
double_jaum_map = {
    'ㄲ': ['ㄱ', 'ㄱ'], 'ㄸ': ['ㄷ', 'ㄷ'], 'ㅃ': ['ㅂ', 'ㅂ'],
    'ㅆ': ['ㅅ', 'ㅅ'], 'ㅉ': ['ㅈ', 'ㅈ'], 'ㄳ': ['ㄱ', 'ㅅ'],
    'ㄵ': ['ㄴ', 'ㅈ'], 'ㄶ': ['ㄴ', 'ㅎ'], 'ㄺ': ['ㄹ', 'ㄱ'],
    'ㄻ': ['ㄹ', 'ㅁ'], 'ㄼ': ['ㄹ', 'ㅂ'], 'ㄽ': ['ㄹ', 'ㅅ'],
    'ㄾ': ['ㄹ', 'ㅌ'], 'ㄿ': ['ㄹ', 'ㅍ'], 'ㅀ': ['ㄹ', 'ㅎ'],
    'ㅄ': ['ㅂ', 'ㅅ']
}

# 겹모음 분해용 매핑
double_moum_map = {
    'ㅘ': ['ㅗ', 'ㅏ'],
    'ㅙ': ['ㅗ', 'ㅐ'],
    'ㅚ': ['ㅗ', 'ㅣ'],
    'ㅝ': ['ㅜ', 'ㅓ'],
    'ㅞ': ['ㅜ', 'ㅔ'],
    'ㅟ': ['ㅜ', 'ㅣ'],
    'ㅢ': ['ㅡ', 'ㅣ']
}

# 전체 자모 목록 구성
all_jamo = sorted(set(jaum_list + moum_list + jaum_list))
char_to_index = {char: idx for idx, char in enumerate(all_jamo)}
index_to_char = {idx: char for char, idx in char_to_index.items()}
VOCAB_SIZE = len(char_to_index)

# 학습시킬 단어
example_sentences = [
    "겠습니다", "이었습니다", "였습니다", "겠다", "이었다", "였다",
    "쌌습니다", "썼습니다", "씻었습니다", "씻습니다", "씁니다", "샀습니다", "쏟았습니다", "쑤셨습니다", "쐈습니다",
    "슈퍼", "슈트", "슈가", "수육", "수유", "수술", "수습", "없음", "없어"
]

# 자모 시퀀스 추출
def decompose_sentence_to_jamo_sequence(sentence):
    result = []
    for char in sentence:
        if hgtk.checker.is_hangul(char):
            try:
                c, v, t = hgtk.letter.decompose(char)

                # 초성 처리
                if c in double_jaum_map:
                    result.extend(double_jaum_map[c])
                else:
                    result.append(c)

                # 중성 처리
                if v in double_moum_map:
                    result.extend(double_moum_map[v])
                else:
                    result.append(v)

                # 종성 처리
                if t != '':
                    if t in double_jaum_map:
                        result.extend(double_jaum_map[t])
                    else:
                        result.append(t)

            except hgtk.exception.NotHangulException:
                continue
    return result

# 전체 시퀀스와 타깃 구성
X_data, y_data = [], []

for sent in example_sentences:
    jamos = decompose_sentence_to_jamo_sequence(sent)
    for i in range(1, len(jamos)):
        input_seq = jamos[:i]
        target = jamos[i]
        X_data.append([char_to_index[j] for j in input_seq])
        y_data.append(char_to_index[target])

# 시퀀스 padding
from tensorflow.keras.preprocessing.sequence import pad_sequences
max_len = 20
X_padded = pad_sequences(X_data, maxlen=max_len, padding='pre')

# 타켓 원핫 인코딩
y_categorical = to_categorical(y_data, num_classes=VOCAB_SIZE)

# 모델 구성
model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=64, input_length=max_len),
    LSTM(128),
    Dense(VOCAB_SIZE, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 모델 학습
model.fit(X_padded, y_categorical, epochs=100, batch_size=8)

# 모델 저장
model.save("SiotYu_distinction.h5")