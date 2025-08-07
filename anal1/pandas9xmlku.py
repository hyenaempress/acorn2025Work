#xml 문서처리

from bs4 import BeautifulSoup

with open('my.xml', 'r', encoding='utf-8') as f:
    xmlfile = f.read()
    #print(xmlfile)

soup = BeautifulSoup(xmlfile, 'lxml')
print(soup)
itemTag = soup.find_All('item')
print(itemTag)
print(itemTag[1])

print()
nameTag = soup.find_All('name')
