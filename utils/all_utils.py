import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap
import joblib
import os

def prepareData(df):
  X = df.drop("y", axis = 1)
  y = df["y"]
  return X,y

def save_model(model, filename):
  model_dir = "models"
  os.makedirs(model_dir, exist_ok= True) #Creates directory only if the directory name is not there
  filepath = os.path.join(model_dir, filename) #models/filename
  joblib.dump(model, filepath)

def save_plot(df, file_name, model):
  def _create_base_plot(df):
    df.plot(x="x1", y="x2", kind="scatter", c="y", s=100, cmap="winter")
    plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
    plt.axvline(x=0, color="black", linestyle="--", linewidth=1)

  def _plot_decission_region(X, y, classifier, resolution=0.02):
    colors = ("red", "blue", "lightgreen")
    cmap = ListedColormap(colors[:len(np.unique(y))])
    X = X.values #as a array
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.2, cmap=cmap)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.plot()


  X,y = prepareData(df)

  _create_base_plot(df)
  _plot_decission_region(X, y, model)

  plot_dir = "plots"
  os.makedirs(plot_dir, exist_ok= True) #Creates directory only if the directory name is not there
  filepath = os.path.join(plot_dir, file_name) #plots/filename
  plt.savefig(filepath)

