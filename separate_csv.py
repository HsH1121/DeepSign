import pandas as pd

# 1. 원본 CSV 로드
df = pd.read_csv("hand_features_with_orientation_with_han.csv", encoding="cp949")

# 2. 라벨 그룹 정의
consonants = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
vowels = ['ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ', 'ㅐ', 'ㅔ', 'ㅖ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅢ']
digits = [str(i) for i in range(1, 11)]  # '1' ~ '10'

# 3. 필터링
df_consonant = df[df['label'].isin(consonants)]
df_vowel = df[df['label'].isin(vowels)]
df_digit = df[df['label'].isin(digits)]

# 4. 파일 저장
df_consonant.to_csv("dataset_consonants.csv", index=False, encoding="cp949")
df_vowel.to_csv("dataset_vowels.csv", index=False, encoding="cp949")
df_digit.to_csv("dataset_digits.csv", index=False, encoding="cp949")

print("✔️ 자음/모음/숫자 데이터셋 분할 및 저장 완료.")
