# XML 문서 처리
from bs4 import BeautifulSoup

with open('./01_big_data_machine_learning/code/0807/my.xml', 'r', encoding='utf-8') as file:
    xml_file = file.read()
    print(xml_file)

soup = BeautifulSoup(xml_file, 'lxml')

# print(soup.prettify())
itemTag = soup.find_all('item')
print("item 태그의 개수:", len(itemTag))

print(itemTag[0])  # 첫 번째 item 태그 출력
print()
nameTag = soup.find_all('name')
print(nameTag[0]['id'])  # 첫 번째 name 태그 출력

print("-------------")

for i in itemTag:
    nameTag = i.find_all('name')
    for j in nameTag:
        print('id:' + j['id'] + ', name:' + j.string)
        tel = i.find('tel')
        print('tel:' + tel.string)
    for j in i.find_all('exam'):
        print('kor:' + j['kor'] + ', eng:' + j['eng'])
    print()