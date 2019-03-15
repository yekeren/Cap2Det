import os
import json

from google.protobuf import text_format
from object_detection.protos import string_int_label_map_pb2

root = 'voc2012.results/per_class_ssquality_12'

with open('configs/voc_vocab.txt', 'r') as fid:
  categories = [x.strip('\n') for x in fid.readlines()]

fids = {}
for category in categories:
  if not os.path.isdir(root + '/results/VOC2012/Main'):
    os.makedirs(root + '/results/VOC2012/Main')
  filename = 'comp3_det_test_%s.txt' % (category)
  filename = os.path.join(root + '/results/VOC2012/Main', filename)
  fids[category] = open(filename, 'w')

count = 0
for filename in os.listdir(root):
  if filename[-5:] == '.json':
    count += 1
    with open(os.path.join(root, filename), 'r') as fid:
      data = json.load(fid)
      for elem in data:
        image_id, category, bbox, score = (elem['image_id'],
                                           elem['category_id'], elem['bbox'],
                                           elem['score'])
        xmin, ymin, width, height = bbox
        xmax, ymax = xmin + width, ymin + height
        line = '%s %.4lf %.1lf %.1lf %.1lf %.1lf\n' % (image_id, score, xmin,
                                                     ymin, xmax, ymax)
        fids[category].write(line)
print('gathered %i results' % (count))
print('done')

for _, fid in fids.items():
  fid.close()
#label_map = string_int_label_map_pb2.StringIntLabelMap()
#with open('configs/mscoco_label_map.pbtxt', 'r') as fid:
#  label_map_string = fid.read()
#  try:
#    text_format.Merge(label_map_string, label_map)
#  except text_format.ParseError:
#    label_map.ParseFromString(label_map_string)
#
#name_to_id = {}
#for item in label_map.item:
#  name_to_id[item.display_name] = item.id
#print(json.dumps(name_to_id, indent=2))
#
#results = []
#for i, filename in enumerate(os.listdir(root)):
#  filename = os.path.join(root, filename)
#  with open(filename, 'r') as fid:
#    data = json.load(fid)
#    for item in data[:100]:
#      item['category_id'] = name_to_id[item['category_id']]
#      results.append(item)
#  if i % 100 ==0:
#    print(i)
#
#with open(result_file, 'w') as fid:
#  fid.write(json.dumps(results))
#print('done')
