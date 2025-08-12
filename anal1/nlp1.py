#위키백과 사이트에서 한번 읽어서 [[KoNLPy]] 참고하여 코드 작성
#pip install konlpy 설치 해야 합니다 

from konlpy.tag  import Okt, Kkma, Komoran, Hannanum

text = "나는 오늘 아침에 강남에 갔다. 가는 길에 빵집이 보여 너무 먹고 싶었다."

okt = Okt()
kkma = Kkma()
komoran = Komoran()
hannanum = Hannanum()

print("=== Okt ===")
print("morphs:", okt.morphs(text))
print("pos:", okt.pos(text))
print("pos(norm, stem):", okt.pos(text, norm=True, stem=True))  # ✅ Okt만 지원
print("nouns:", okt.nouns(text))

print("\n=== Kkma ===")
print("morphs:", kkma.morphs(text))
print("pos:", kkma.pos(text))          # ❌ norm/stem 미지원
print("nouns:", kkma.nouns(text))

print("\n=== Komoran ===")
print("morphs:", komoran.morphs(text))
print("pos:", komoran.pos(text))       # ❌ norm/stem 미지원
print("nouns:", komoran.nouns(text))

print("\n=== Hannanum ===")
print("morphs:", hannanum.morphs(text))
print("pos:", hannanum.pos(text))
print("nouns:", hannanum.nouns(text))
komoran = Komoran()


print('택스트 2')
#pip install wordcloud 설치 해야 합니다 

from wordcloud import WordCloud
import matplotlib.pyplot as plt

text2 = "나는 오늘 아침에 강남에 갔다. 가는 길에 강남에 있는 빵집이 보여 너무 먹고 싶었다. 빵이 특히 강남에 있는"

nouns = okt.nouns(text2)
words = " ".join(nouns)
print("words: ", words)

wc = WordCloud(font_path="malgun.ttf", width=400, height=300, background_color="white")
cloud = wc.generate(words)
plt.imshow(cloud, interpolation="bilinear")
plt.axis("off")
plt.show()
