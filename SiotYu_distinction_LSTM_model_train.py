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
    "갔습니다", "썼습니다", "쐈습니다", "쌓았습니다", "씻었습니다", "쌌습니다", "넣었습니다", "받았습니다",
    "말했습니다", "기록했습니다", "제출했습니다", "표현했습니다", "설명했습니다", "확인했습니다",
    "시작했습니다", "완료했습니다", "참여했습니다", "감사했습니다", "사과했습니다", "받쳐썼습니다",
    "추천했습니다", "지적했습니다", "분석했습니다", "경고했습니다", "답변했습니다",
    "쐈었네요", "썼었어요", "쌌었네요", "썼겠지요", "썼었겠죠", "쐈겠네요", "쌌겠네요", "쐈겠어요",
    "쌓겠어요", "썼겠어요", "썼을걸요", "쌌을걸요", "쐈을걸요", "썼을까요", "쐈을까요", "쌌을까요",
    "하겠습니다", "말씀드리겠습니다", "도와드리겠습니다", "출발하겠습니다", "준비하겠습니다",
    "이해하겠습니다", "전달하겠습니다", "기록하겠습니다", "끝내겠습니다", "진행하겠습니다",
    "소개하겠습니다", "공유하겠습니다", "복습하겠습니다", "해결하겠습니다", "보완하겠습니다",
    "정리하겠습니다", "탐색하겠습니다", "참석하겠습니다", "도전하겠습니다", "배우겠습니다",
    "학생이었습니다", "선생님이었습니다", "의사였습니다", "배우였습니다", "가수였습니다",
    "친구였습니다", "형이었습니다", "누나였습니다", "엄마였습니다", "아빠였습니다",
    "운전자였습니다", "직원였습니다", "경찰이었습니다", "의뢰인이었습니다", "지원자였습니다",
    "환자였습니다", "도전자였습니다", "후보였습니다", "관리자였습니다", "사장님이었습니다",
    "휴가입니다", "유학입니다", "유리였습니다", "유전자입니다", "유산소입니다",
    "유치원입니다", "유행이었습니다", "유쾌했습니다", "유익했습니다", "유일했습니다",
    "유도하였습니다", "유지하였습니다", "유산남겼습니다", "유머였습니다", "유료였습니다",
    "유죄였습니다", "유사했습니다", "유유상종입니다", "휴식을취했습니다", "휴대폰입니다",
    "아팠어요ㅠㅠ", "힘들었어요ㅠㅠ", "속상했어요ㅠㅠ", "화났어요ㅠㅠ", "울었어요ㅠㅠ",
    "떨어졌어요ㅠㅠ", "실망했어요ㅠㅠ", "서운했어요ㅠㅠ", "외로웠어요ㅠㅠ", "슬펐어요ㅠㅠ",
    "부끄러웠어요ㅠㅠ", "실수했어요ㅠㅠ", "늦었어요ㅠㅠ", "잊었어요ㅠㅠ", "무서웠어요ㅠㅠ",
    "봤습니다", "들었습니다", "읽었습니다", "적었습니다", "먹었습니다", "마셨습니다",
    "찍었습니다", "탔습니다", "앉았습니다", "일어났습니다", "열었습니다", "닫았습니다",
    "살았습니다", "웃었습니다", "울었습니다", "찾았습니다", "걸었습니다", "준비했습니다", "운전했습니다", "기다렸습니다",
    "이해했습니다", "받아썼습니다", "되돌아봤습니다", "참고했습니다", "약속했습니다",
    "감상했습니다", "계획했습니다", "반성했습니다", "회상했습니다", "기록했습니다",
    "조사했습니다", "처리했습니다", "응답했습니다", "전송했습니다", "복습했습니다",
    "유도했습니다", "유지했습니다", "유추했습니다", "유입되었습니다", "유래되었습니다",
    "유출되었습니다", "유입경로입니다", "유기농입니다", "유산소운동했습니다", "유산남겼습니다",
    "유능했습니다", "유사품입니다", "유명했습니다", "유용했습니다", "유일무이했습니다",
    "기숙사였습니다", "미숙했습니다", "휴지통입니다", "규칙지켰습니다", "교육받았습니다",
    "숙지했습니다", "수집했습니다", "수정했습니다", "구입했습니다", "기운없었습니다",
    "충실했습니다", "준수했습니다", "시작했습니다", "준비됐습니다", "출석했습니다"
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
                    if c != '':
                        result.append(c)

                # 중성 처리
                if v in double_moum_map:
                    result.extend(double_moum_map[v])
                else:
                    if v != '':
                        result.append(v)

                # 종성 처리
                if t != '':
                    if t in double_jaum_map:
                        result.extend(double_jaum_map[t])
                    else:
                        if t != '':
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
model.save("SiotYu_distinction_2.h5")