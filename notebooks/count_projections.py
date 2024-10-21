"""
A simple file to count how many projections exist in a dataset. This is how we got the 2.50 number for EMNLP submission.
"""

import json
from tqdm import tqdm

with open('/home/oval/wikidata-dataset/new_dataset/dev_cleaned.json', "r") as f:
    d = json.load(f)
print(len(d))

with open('/home/oval/wikidata-dataset/new_dataset/test_0615_4am_cleaned.json', "r") as f:
    temp=json.load(f)
    print(len(temp))
    d += temp


# for getting how many projects exist, we simply go with executing gold query since that is easier
import sys
sys.path.append('/home/oval/wikidata-dataset')
from wikidata_utils import execute_sparql

count = 0
projections = 0
for i in tqdm(d):
    sparql_result = execute_sparql(i["sparql"])
    if sparql_result == []:
        continue
    
    count += 1
    projections += len([i for i in sparql_result[0].keys() if not (i.endswith('Label') or i.endswith('Description'))])
    
print(projections/count)