import os
import json

weights = json.load(open('weights.json'))
new_weights = weights[2:-6]

json.dump(new_weights, open('weights.json', 'w'))

