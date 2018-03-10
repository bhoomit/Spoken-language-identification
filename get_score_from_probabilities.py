""" USAGE: python get_score_from_probabilities.py --prediction= --anwser=
    prediction file may have less lines
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--prediction', type=str)
args = parser.parse_args()
print(args)

prediction_file = open(args.prediction, 'r')
prediction_lines = prediction_file.readlines()
cnt = len(prediction_lines)

for iter in range(cnt):
    out = prediction_lines[iter].split(',')
    out = [float(x) for x in out]
    pred = [(x, it) for it, x in enumerate(out)]
    pred = sorted(pred, reverse=True)
    print(pred[0][1])
