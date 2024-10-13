import csv
import re
from googletrans import Translator

translator = Translator()

def translate(s):
    return translator.translate(s, dest="en").text

with open('qcm_ref.csv', "r") as infile, open('qcm_ref_en.csv', "w") as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        translated_row = [translate(item) for item in row]
        writer.writerow(translated_row)