# JSON: XML의 비해 가벼우며, 배열에 대한 지식만 있으면 처리 가능
import json
import urllib.request as req
import pandas as pd

dict = {
    'name':'tom', 
    'age':33, 
    'score':['90', '80', '100']
    }

print("dict:%s"%dict)
print(type(dict))

print('json encoding (dict를 JSON 모양의 문자열로 변경하는 것)---')
str_val = json.dumps(dict)
print("str_val:%s"%str_val)
print(type(str_val))
# print(str_val['name']) # 오류 발생, str_val은 문자열이므로 인덱싱 불가

print('json decoding (JSON 모양의 문자열을 dict로 변경하는 것)---')
dict2 = json.loads(str_val)
print("dict2:%s"%dict2)
print(type(dict2))
print(dict2['name'])  # 인덱싱 가능 

# dict2이므로 가능
for key in dict2.keys():
    print("key:%s, value:%s"%(key, dict2[key]))

print('웹에서 JSON 문서 읽기')
url = 'https://raw.githubusercontent.com/pykwon/python/master/seoullibtime5.json'
plainText = req.urlopen(url).read().decode('utf-8')
print("plainText:%s"%plainText)
print(type(plainText))
jsonData = json.loads(plainText)
print("jsonData:%s"%jsonData)
print(type(jsonData))

print(jsonData['SeoulLibraryTime']['row'][0]['LBRRY_NAME'])  # 인덱싱 가능

# dict의 자료를 읽어 도서관명, 전화, 주소를 출력
libData = jsonData.get('SeoulLibraryTime').get('row')

# print(libData)
print(libData[0].get('LBRRY_NAME'))

datas = []
for ele in libData:
    name = ele.get('LBRRY_NAME')
    tel = ele.get('TEL_NO')
    addr = ele.get('ADRES')
    # print(name + '\t' + tel + '\t' + addr)
    datas.append([name, tel, addr])

print(datas)
df = pd.DataFrame(datas, columns=['도서관명', '전화', '주소'])
print(df)


