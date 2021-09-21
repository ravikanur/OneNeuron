from utils.model import  Perceptron
from utils.all_utils import prepareData
import pandas as pd
AND = {
    "x1":[0,0,1,1],
    "x2":[0,1,0,1],
    "y":[0,0,0,1]
}

df_AND = pd.DataFrame(AND)
X,y = prepareData(df_AND)

model_AND = Perceptron(eta=ETA, epochs=EPOCHS)
model_AND.fit(X,y)