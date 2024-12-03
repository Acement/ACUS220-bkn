import os


csv_dir = '/home/pablo/Desktop/ACUS220/ESC-50/meta/esc50.csv' #Directorio del csv con los tags para el modelo
csv_spec_dir = '/home/pablo/Desktop/ACUS220/ACUS220-bkn/data/spec_csv.csv'

file_read = open(csv_dir,'r')

file_write = []

print('DATA:')

for i in file_read:
    print(i)
    temp_line = i.replace('wav','png')
    file_write.append(temp_line)

with open(csv_spec_dir,'a+') as f:
    for i in range(0,len(file_write)):
        print(file_write[i])
        f.write(file_write[i])

