
import os
import json

from google.protobuf import text_format
from object_detection.protos import string_int_label_map_pb2

root = 'coco.results/per_class_ssquality_coco17_cap_learned_w2v_match'
result_file = 'testdev_learned_w2v_match.json'

label_map = string_int_label_map_pb2.StringIntLabelMap()
with open('configs/mscoco_label_map.pbtxt', 'r') as fid:
  label_map_string = fid.read()
  try:
    text_format.Merge(label_map_string, label_map)
  except text_format.ParseError:
    label_map.ParseFromString(label_map_string)

name_to_id = {}
for item in label_map.item:
  name_to_id[item.display_name] = item.id
print(json.dumps(name_to_id, indent=2))

results = []
for i, filename in enumerate(os.listdir(root)):
  filename = os.path.join(root, filename)
  with open(filename, 'r') as fid:
    data = json.load(fid)
    for item in data[:100]:
      item['category_id'] = name_to_id[item['category_id']]
      results.append(item)
  if i % 100 ==0:
    print(i)

with open(result_file, 'w') as fid:
  fid.write(json.dumps(results))
print('done')
