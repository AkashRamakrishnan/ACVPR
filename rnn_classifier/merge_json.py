import os
import json

data_dir = 'final_data'
output_file = 'merge.json'

data = []
class_list = []
for fname in os.listdir(data_dir):
    if fname.endswith('.json'):
        print('y')
        with open(os.path.join(data_dir, fname)) as f:
            class_data = json.load(f)
            # print(len(class_data))
            num = len(class_data) 
            class_name = fname.split('.')[0]
            data = data + class_data
            class_list = class_list + [class_name]*num

final_dic = {'class_list':class_list, 'features':data}
with open(output_file, 'w') as f:
    json.dump(final_dic, f, indent = 6)
    