#JSON 특징 , XML 에 비해 가벼우며 배열 형식으로 데이터를 표현한다. 
#xml 보다 가볍고  표준이라 JASAN 형식을 많이 쓴다. 
#json 형식은 파이썬에서 딕셔너리 형식으로 쉽게 변환할 수 있다. 
#딕트랑 궁합이 잘맞아요

import json

dict = {'name': 'tom', 'age': 33, 'score': ['90', '80', '100']} 
print(dict)
print("dict:%s" % dict)
print(type(dict)) # 딕트 타입 

print('json 인코딩--  딕트를 제이슨 모양의 문자열로 변경하는 것을 제이슨 인코딩이라고 한다')

str_val = json.dumps(dict) #이렇게 되면 딕트를 제이슨 인코딩 
print("str_val:%s" % str_val)
print(type(str_val)) # 제이슨 타입  
#이제 문자열이 된다 
#print(str_val['name']) #타입 에러 발생
#딕트를 문자열로 바꿀땐 덤스 반대의 경우는 디코드 
print('json 디코딩--  제이슨 문자열을 딕트 형식으로 변경하는 것을 제이슨 디코딩이라고 한다')
jason_val = json.loads(str_val)
print(jason_val)
print(type(jason_val))
print(jason_val['name']) #타입 에러가 발생하지 않는다. 이런식으로 디코딩 인코딩이 가능하다

for k in jason_val.keys():
    print(k) # 이건 딕트이니까 가능합니다. JSON 

print('웹에서 제이슨 문서 읽기 ==========================')

import urllib.request # as req 를 넣으면 req 로 줄여서 사용할 수 있다. 

url = "https://raw.githubusercontent.com/pykwon/python/master/seoullibtime5.json"
    # 읽어온 데이터를 제이슨 데이터를 딕트로 변환 

response = urllib.request.urlopen(url) 
plainText = response.read().decode('utf-8')    
json_data = json.loads(plainText)  # 읽어온 데이터를 제이슨 형식으로 변환 


print(json_data)
print(type(json_data))
#print(json_data['data'][0]['TIME'])
#print(json_data['seoulLibraryTime']['row'][0]['LBRRY_NAME'])

#dict 의 자료를 읽어 도서관명, 전화, 주소  

"""
libData = json_data['SeoulLibraryTime']['row']
for lib in libData:
    print(lib['LBRRY_NAME'], lib['TEL'], lib['ADRES'])

"""

libData = json_data.get('seoulLibraryTime').get('row')
# print(libData)
print(libData[0].get('LBRRY_NAME'))
for ele in libData:
    print(ele.get('LBRRY_NAME'), ele.get('TEL'), ele.get('ADRES'))

#데이터 프레임으로 변환 
import pandas as pd
df = pd.DataFrame(libData, columns=['LBRRY_NAME', 'TEL', 'ADRES'])
print(df)
df.to_csv('seoulLibrary.csv', index=False, encoding='utf-8')
        
#얼른 이거 마무리하고 시각화 맥플로립으로 갈게요 ~ 




