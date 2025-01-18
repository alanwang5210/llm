import json

with open('C:\\Users\\10100\\Downloads\\response.json', 'r',encoding="UTF-8") as file:
    data = json.load(file)

print(data)