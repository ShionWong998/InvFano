import csv
import ast
def findeps(filename,f):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == f:
                col2 = ast.literal_eval(row[1].strip())
                col3 = ast.literal_eval(row[2].strip())
                col4 = ast.literal_eval(row[3].strip())
                return col2,col3,col4
    return None,None,None