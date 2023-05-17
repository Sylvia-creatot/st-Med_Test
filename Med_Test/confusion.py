import pandas as pd

data = {'y_actual':    [1,0,1,0,0,0,0,0,0,],
        'y_predicted': [0,0,1,0,0,0,0,0,0]
        }

df = pd.DataFrame(data)

confusion_matrix = pd.crosstab(df['y_actual'], df['y_predicted'], rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)