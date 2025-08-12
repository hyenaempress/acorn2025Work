#위키백과 사이트에서 한번 읽어서 [[KoNLPy]] 참고하여 코드 작성
#pip install konlpy 설치 해야 합니다 

from konlpy.tag  import Okt, Kkma, Komoran, Hannanum

# Corpus 우리말로는 말뭉치 : 자연어 처리를 목적으로 수집된 문장 집단 
text = "나는 오늘 아침에 강남에 갔다. 가는 길에 빵집이 보여 너무 먹고 싶었다."
#연습용 텍스트 -> corpus 

print("Okt 형태소 분석: ")
okt = Okt() #객체를 생성합니다
print(okt.morphs(text)) #형태소 분석 
#형태소 분석 라이브러리로 많이 쓰는게 코넬 파이 

print('품사 태깅: ', okt.pos(text)) #품사 태깅 
print('품사 태깅 (어간 포함):', okt.pos(text, norm=True, stem=True)) #원형 어근 포함 
print('명사 추출: ', okt.nouns(text)) #명사 추출 

print('Kkma 형태소 분석: ')
kkma = Kkma()
print(kkma.morphs(text))
print('품사 태깅: ', kkma.pos(text))
print('품사 태깅 (어간 포함):', kkma.pos(text, norm=True, stem=True))
print('명사 추출: ', kkma.nouns(text))

print('Komoran 형태소 분석: ')
komoran = Komoran()
print(komoran.morphs(text))
print('품사 태깅: ', komoran.pos(text))
print('품사 태깅 (어간 포함):', komoran.pos(text, norm=True, stem=True))
print('명사 추출: ', komoran.nouns(text))

komoran = Komoran()


print('택스트 2')
#pip install wordcloud 설치 해야 합니다 

from wordcloud import WordCloud
import matplotlib.pyplot as plt

text2 = "나는 오늘 아침에 강남에 갔다. 가는 길에 강남에 있는 빵집이 보여 너무 먹고 싶었다. 빵이 특히 강남에 있는"

nouns = komoran.nouns(text2)
words = "".join(nouns)
print(words)

wc= WordCloud(font_path='C:/Windows/Fonts/malgun.ttf', width=400, height=300, scale=2.0, max_words=2000, background_color='white')

