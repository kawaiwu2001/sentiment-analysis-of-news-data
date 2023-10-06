import json
import sys
import pandas as pd

from google_trans_new import google_translator


def trans(text):
    lang = "en"
    t = google_translator(timeout=5)
    if len(text) > 5000:
        index = text.index(', ', 3000,5000)
        if index == -1:
            index = text.index('. ', 4000,5000)
        text1 = text[0:index]
        text2 = text[index:]
        translate_text = t.translate(text1, lang)
        translate_text += t.translate(text2, lang)
    else:
        translate_text = t.translate(text, lang)
    return translate_text


with open("test.json", 'r', encoding="UTF-8") as f:
    test_data = json.load(f)

for cont in test_data:
    cont['content'] = trans(cont['content'])
    print(cont['content'])

with open('new_data_test.json', 'w') as f:
    json.dump(test_data, f)

with open("train.json", 'r', encoding="UTF-8") as f:
    train_data = json.load(f)

for cont in train_data:
    cont['content'] = trans(cont['content'])
    print(cont['content'])

with open('new_data_train.json', 'w') as f:
    json.dump(train_data, f)

