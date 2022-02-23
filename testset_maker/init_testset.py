import csv

# Run init_testset.py to get `NUM_LABELS` of each label type
# from `IN_PATH` and add it to the test set at `OUT_PATH`.

IN_PATH = '../datasets/seattle_crops.csv'
OUT_PATH = '../datasets/test_set.csv'
NUM_LABELS = 1000

csv_in = open(IN_PATH, 'r')
csv_reader = csv.reader(csv_in)
next(csv_reader)

csv_out = open(OUT_PATH, 'w')
csv_writer = csv.writer(csv_out)
csv_writer.writerow(['image_name', 'label_type'])

counts = [0, 0, 0, 0]
for row in csv_reader:
  filename = row[0]
  label_id = int(row[1])
  if counts[label_id - 1] < NUM_LABELS:
    counts[label_id - 1] += 1
    csv_writer.writerow([filename, [label_id]])
  if counts[0] == NUM_LABELS and counts[1] == NUM_LABELS and counts[2] == NUM_LABELS and counts[3] == NUM_LABELS:
    break
if counts[0] == NUM_LABELS and counts[1] == NUM_LABELS and counts[2] == NUM_LABELS and counts[3] == NUM_LABELS:
  print('Success!')
else:
  print('Was not able to fetch 1000 of each label type! Counts: ', counts)
