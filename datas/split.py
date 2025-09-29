import csv
import os

example_path = 'shape.csv'
result_dir = './shapes/'

with open(example_path, 'r', newline='') as example_file:
    example = csv.reader(example_file, delimiter=',')
    i = j = 1
    for row in example:
        #print(row)
        if i % 5000 == 0:
            j += 1
        result_path = result_dir + str(j) + '.csv'
        print(result_path)

        if not os.path.exists(result_path):
            with open(result_path, 'w', newline='') as file:
                csvwriter = csv.writer(file)
                csvwriter.writerow(row)
            i += 1

        else:
            with open(result_path, 'a', newline='') as file:
                csvwriter = csv.writer(file)
                csvwriter.writerow(row)
            i += 1