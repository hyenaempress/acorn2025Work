# pip install konlpy

from konlpy.tag import Okt, Kkma, Komoran
from matplotlib.pyplot import stem
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# corpus(말뭉치) : 자연어 처리를 목적으로 수집된 문자 집단
text1 = "나는 오늘 아침에 강남에 갔다. " \
"가는 길에 빵집이 보여 너무 먹고 싶었다. " \
"나는 배고프다. 나는 너무 배고파서 돈까스가 먹고 싶다. " \
"나는 스윙스처럼 돈까스가 많이 먹고 싶다. " \
"나는 배고프다. " \
"근데 살을 빼야 한다."

# 형태소: 최소 의미 단위

# print("Okt----------------------------------------")
okt = Okt()
print("형태소: ", okt.morphs(text1))
print("품사 태깅: ", okt.pos(text1))
print("품사 태깅(어간 포함): ", okt.pos(text1, stem=True)) # 원형(어근)으로 출력. 그래요. -> 그렇다
print("명사 추출: ", okt.nouns(text1))

# print("Kkma----------------------------------------")
# kkma = Kkma()
# print("형태소: ", kkma.morphs(text1))
# print("품사 태깅: ", kkma.pos(text1))
# print("명사 추출: ", kkma.nouns(text1))

# print("Komoran----------------------------------------")
# komoran = Komoran()
# print("형태소: ", komoran.morphs(text1))
# print("품사 태깅: ", komoran.pos(text1))
# print("명사 추출: ", komoran.nouns(text1))


text2 = "나는 오늘 아침에 강남에 갔다. " \
"가는 길에 빵집이 보여 너무 먹고 싶었다." \
"나는 배고프다. 나는 너무 배고파서 돈까스가 먹고 싶다. " \
"나는 스윙스처럼 돈까스가 많이 먹고 싶다. " \
"나는 배고프다. " \
"근데 살을 빼야 한다." \
"그치만 돈까스가 먹고 싶다."

nouns = okt.nouns(text2)
words = " ".join(nouns)
print("words: ", words)

wc = WordCloud(font_path="malgun.ttf", width=400, height=300, background_color="white")
cloud = wc.generate(words)
plt.imshow(cloud, interpolation="bilinear")
plt.axis("off")
plt.show()
