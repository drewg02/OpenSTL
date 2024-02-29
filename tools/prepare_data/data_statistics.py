import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time
import argparse
import os
import itertools
import plotly.graph_objects as go
import pandas as pd


def combine_data(X, Y):
    combined_data = np.concatenate((X, Y), axis=1)
    combined_data = combined_data.reshape(-1, 64, 64)
    return combined_data

def calculate_stats(data):
    means = np.mean(data, axis=(0, 2))
    stds = np.std(data, axis=(0, 2))
    return means, stds

def visualize(method, data, title, save_path, n_components=2, save_data=False, size=20):
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    if n_components == 3:
        fig_combined = plt.figure(figsize=(10, 7))
        ax_combined = fig_combined.add_subplot(111, projection='3d')
    else:
        fig_combined = plt.figure(figsize=(10, 6))

    transformed_data_save = {}
    for i, (name, dataset) in enumerate(data):
        reshaped_data = dataset.reshape(-1, dataset.shape[-1] * dataset.shape[-2])
        color = next(color_cycle)

        if method == 'TSNE':
            model = TSNE(n_components=n_components)
        elif method == 'PCA':
            model = PCA(n_components=n_components)
        transformed_data = model.fit_transform(reshaped_data)

        if n_components == 3:
            ax_combined.scatter(transformed_data[:, 0], transformed_data[:, 1], transformed_data[:, 2], alpha=0.5, label=name, color=color, s=size)
        else:
            fig_combined.gca().scatter(transformed_data[:, 0], transformed_data[:, 1], alpha=0.5, label=name, color=color, s=size)

        transformed_data_save[name] = transformed_data
        
        if n_components == 3:
            fig_individual = plt.figure(figsize=(8, 7))
            ax_individual = fig_individual.add_subplot(111, projection='3d')
            ax_individual.scatter(transformed_data[:, 0], transformed_data[:, 1], transformed_data[:, 2], alpha=0.5, color=color, s=size)
        else:
            fig_individual = plt.figure(figsize=(8, 6))
            plt.gca().scatter(transformed_data[:, 0], transformed_data[:, 1], alpha=0.5, color=color, s=size)

        plt.title(f'{title} - {name}')
        plt.savefig(f'{save_path}_{name}.png')
        plt.close(fig_individual)
    
    if n_components == 3:
        ax_combined.set_title(title)
        ax_combined.legend()
    else:
        fig_combined.gca().set_title(title)
        fig_combined.gca().legend()
    fig_combined.savefig(f'{save_path}.png')
    plt.close(fig_combined)
    
    if n_components == 3:
        df_train = pd.DataFrame(data=transformed_data_save['train'], columns=['PC1', 'PC2', 'PC3'])
        df_val = pd.DataFrame(data=transformed_data_save['val'], columns=['PC1', 'PC2', 'PC3'])
        df_test = pd.DataFrame(data=transformed_data_save['test'], columns=['PC1', 'PC2', 'PC3'])

        color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

        fig = go.Figure()
        
        plotly_size = int(size * (6 / 20))
        fig.add_scatter3d(x=df_train['PC1'], y=df_train['PC2'], z=df_train['PC3'], mode='markers', marker=dict(color=next(color_cycle), size=plotly_size), name='train')
        fig.add_scatter3d(x=df_val['PC1'], y=df_val['PC2'], z=df_val['PC3'], mode='markers', marker=dict(color=next(color_cycle), size=plotly_size), name='val')
        fig.add_scatter3d(x=df_test['PC1'], y=df_test['PC2'], z=df_test['PC3'], mode='markers', marker=dict(color=next(color_cycle), size=plotly_size), name='test')

        fig.update_layout(title=f'{title} - {name}')

        fig.write_html(f'{save_path}.html')
    
    if save_data:
        with open(f'{save_path}.pkl', 'wb') as f:
            pickle.dump(transformed_data_save, f)
    
def shuffle(datasets):
    for i, (name, data) in enumerate(datasets):
        np.random.shuffle(data)
        datasets[i] = (name, data)
    return datasets
    
def run(dataset_path, folder, dataset_name):
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    train_data = combine_data(data['X_train'], data['Y_train'])
    val_data = combine_data(data['X_val'], data['Y_val'])
    test_data = combine_data(data['X_test'], data['Y_test'])

    train_means, train_stds = calculate_stats(train_data)
    val_means, val_stds = calculate_stats(val_data)
    test_means, test_stds = calculate_stats(test_data)
    
    rows = np.arange(64) 

    plt.figure()
    plt.plot(rows, train_means, label='Train Mean')
    plt.plot(rows, val_means, label='Val Mean')
    plt.plot(rows, test_means, label='Test Mean')
    plt.xlabel('Row')
    plt.ylabel('Mean')
    plt.title(f'Mean - {dataset_name}')
    plt.legend()
    plt.ylim(0, 1)
    plt.savefig(f'{folder}/{dataset_name}_mean_comparison.png')
    
    plt.figure()
    plt.plot(rows, train_stds, label='Train Std')
    plt.plot(rows, val_stds, label='Val Std')
    plt.plot(rows, test_stds, label='Test Std')
    plt.xlabel('Row')
    plt.ylabel('Standard Deviation')
    plt.title(f'Std dev - {dataset_name}')
    plt.legend()
    plt.ylim(0, 1)
    plt.savefig(f'{folder}/{dataset_name}_std_comparison.png')
    
    datasets = [('train', train_data), ('val', val_data), ('test', test_data)]
    visualize('TSNE', datasets, f't-SNE - {dataset_name}', f'{folder}/{dataset_name}_tsne', size=10)
    visualize('PCA', datasets, f'PCA - {dataset_name}', f'{folder}/{dataset_name}_pca', size=10)
    visualize('TSNE', datasets, f't-SNE - {dataset_name}', f'{folder}/{dataset_name}_3components_tsne', n_components=3, save_data=True, size=10)
    visualize('PCA', datasets, f'PCA - {dataset_name}', f'{folder}/{dataset_name}_3components_pca', n_components=3, save_data=True, size=10)
    #if dataset_name == "dataset_e3dlstm_2dplate_1000plates_0.1chance_10-10_64x64_v1_random":
    #    for i in range(24):
    #        datasets = shuffle(datasets)
    #        plot_visualization('TSNE', datasets, f't-SNE - {dataset_name}', f'{folder}/{dataset_name}_tsne_shuffle{i}.png')
    #        plot_visualization('PCA', datasets, f'PCA - {dataset_name}', f'{folder}/{dataset_name}_pca_tsne_shuffle{i}.png')
    #        if dataset_name == "dataset_e3dlstm_2dplate_1000plates_0.1chance_10-10_64x64_v1_random":
    #            plot_visualization('TSNE', datasets, f't-SNE - {dataset_name}', f'{folder}/{dataset_name}_3components_tsne_shuffle{i}.png', n_components=3)
    #            plot_visualization('PCA', datasets, f'PCA - {dataset_name}', f'{folder}/{dataset_name}_3components_pca_shuffle{i}.png', n_components=3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process multiple datasets.")
    parser.add_argument("dataset_paths", nargs='+', type=str, help="Dataset names")
    args = parser.parse_args()
    
    for dataset_path in args.dataset_paths:
        folder, dataset_name = dataset_path.rsplit('/', 1)
        dataset_name = dataset_name.rsplit('.', 1)[0]
        
        print(f'Starting run for dataset {dataset_name}')
        start_time = time.time()
        
        run(dataset_path, folder, dataset_name)
        
        print(f'Finished run for dataset {dataset_name}, took {time.time() - start_time} seconds')
