import pickle
#강사님이 피클 좋아하셔서 피클로 하지만 따른거 해도 됩니다.

config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '970506',
    'db': 'mydb',
    'port': 3306,
    'charset': 'utf8'
} #이런식으로 딕트로 만들어서 넣어 줄 수 있습니다. 


with open('mymaria.dat', 'wb') as obj:
    pickle.dump(config, obj)

print("mymaria.dat 파일이 생성되었습니다.")