import json
from collections import Counter

_UNCLEAR = 'unclear'


def _load_topics(topic_json_input_path, topic_namelist_txt_input_path):
  """Loads topic annotations.

  Args:
    topic_json_input_path: Path to the directory saving topic annotations.
    topic_namelist_txt_input_path: Path to the file storing topic list.

  Returns:
    A python dict mapping from image ID to the topic annotations.
  """
  with open(topic_namelist_txt_input_path, 'r') as fid:
    topic_list = [line.strip('\n') for line in fid.readlines()]

  topics = dict(
      [(str(index + 1), topic) for index, topic in enumerate(topic_list)])

  topic_annotations = {}
  with open(topic_json_input_path, 'r') as fid:
    data = json.load(fid)
    for image_id, annotations in data.items():
      image_id = int(image_id.split('/')[1].split('.')[0])
      counter = Counter([
          topics.get(annotation, _UNCLEAR)
          for annotation in annotations
          if annotation in topics
      ])
      if len(counter):
        topic = counter.most_common()[0][0]
      else:
        topic = _UNCLEAR
      topic_annotations.setdefault(topic, []).append(image_id)
  return topic_list, topic_annotations


topic_list, topic_data = _load_topics(
    'raw_data/ads_test_annotations/Topics_test.json',
    'raw_data/topics_list.txt')

with open('topic_to_image_ids.json', 'w') as fid:
  fid.write(json.dumps(topic_data, indent=2))
