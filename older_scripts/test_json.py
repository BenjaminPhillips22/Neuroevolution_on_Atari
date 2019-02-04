import json
import sys

f_name = sys.argv[1]

d = json.load(open(f_name))
# d = pd.read_json('test.json')
print(d)
print(type(d))
print(d['name'])


# need to close the opened file