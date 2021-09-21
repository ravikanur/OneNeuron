from utils.model import  Perceptron
from utils.all_utils import prepareData, save_model, save_plot
import pandas as pd

AND = {
    "x1":[0,0,1,1],
    "x2":[0,1,0,1],
    "y":[0,0,0,1]
}

df_AND = pd.DataFrame(AND)
X,y = prepareData(df_AND)
ETA = 0.3
EPOCHS = 10

model_AND = Perceptron(eta=ETA, epochs=EPOCHS)
model_AND.fit(X,y)

save_model(model_AND, "AND.model")

save_plot(df_AND, "AND.png", model_AND)