import pandas as pd
import plotly.graph_objects as go
import pickle
import argparse
import itertools
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and display a 3D PCA plot.")
    parser.add_argument("pca_path", type=str, help="PCA Path")
    args = parser.parse_args()

    with open(args.pca_path, 'rb') as f:
        data = pickle.load(f)

    df_train = pd.DataFrame(data=data['train'], columns=['PC1', 'PC2', 'PC3'])
    df_val = pd.DataFrame(data=data['val'], columns=['PC1', 'PC2', 'PC3'])
    df_test = pd.DataFrame(data=data['test'], columns=['PC1', 'PC2', 'PC3'])

    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    fig = go.Figure()

    fig.add_scatter3d(x=df_train['PC1'], y=df_train['PC2'], z=df_train['PC3'], mode='markers', marker=dict(color=next(color_cycle)), name='train')
    fig.add_scatter3d(x=df_val['PC1'], y=df_val['PC2'], z=df_val['PC3'], mode='markers', marker=dict(color=next(color_cycle)), name='val')
    fig.add_scatter3d(x=df_test['PC1'], y=df_test['PC2'], z=df_test['PC3'], mode='markers', marker=dict(color=next(color_cycle)), name='test')

    fig.update_layout(title='3D PCA Plot')

    fig.write_html('plot.html')
