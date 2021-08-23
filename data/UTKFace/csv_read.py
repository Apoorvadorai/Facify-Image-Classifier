import csv
import spdlog
import os
log = spdlog.ConsoleLogger("csv")

field_names = ['image_path','age','gender','ethnicity']
row = {}
# csv_w = open("./newattributes.csv", 'w')
folderpath = '/home/server1/Documents/alina/PyTorch-Multi-Label-Image-Classification/data/UTKFace/images/'
buckets = ['u10', 'u20', 'u40', 'u60', 'a60']

with open("./newattributes.csv", 'w') as csv_w:
    csv_writer = csv.DictWriter(csv_w, fieldnames=field_names)
    csv_writer.writeheader()
    for ofile in os.listdir(folderpath):
        file = ofile.split('_')
        if len(file) != 4:
            # log.warn("Path = images/{}", str(ofile))
            print(ofile)
            continue

        age = int(file[0])
        if age > 0 and age <= 10:
            age = 'u10'
        elif age >10 and age <= 20:
            age = 'u20'
        elif age > 20 and age <= 40:
            age = 'u40'
        elif age > 40 and age <= 60:
            age = 'u60'
        else:
            age = 'a60'

        gender = 'male' if int(file[1]) == 0 else 'female'

        eth = int(file[2])
        if eth == 0:
            eth = 'white'
        elif eth == 1:
            eth = 'black'
        elif eth == 2:
            eth = 'asian'
        elif eth == 3:
            eth = 'indian'
        else:
            eth = 'others'
        
        row['image_path'] = folderpath+ofile
        row['age'] = age
        row['gender'] = gender
        row['ethnicity'] = eth

        csv_writer.writerow(row)        
        # log.info(f"path = {row['image_path']}, age = {age}, gender = {gender}, ethnicity = {eth}" )