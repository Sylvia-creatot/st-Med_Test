import matplotlib.pyplot as plt
import csv
  
Disease = []
Symptom = []
  
with open('Training2.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines:
        Disease.append(row[0])
        Symptom.append(str(row[1]))
  
plt.scatter(Disease, Symptom, color = 'g',s = 100)
plt.xticks(rotation = 25)
plt.xlabel('Disease')
plt.ylabel('Symptom')
plt.title('Women STD', fontsize = 20)
  
plt.show()