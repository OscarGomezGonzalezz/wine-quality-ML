from clean import clean_dataset 
from exploration import explore
from model import model
# Clean and save multiple datasets
df1=clean_dataset("data/winequality-red.csv")
df2=clean_dataset("data/winequality-white.csv")

# explore(df1)
# explore(df2)
model(df1)