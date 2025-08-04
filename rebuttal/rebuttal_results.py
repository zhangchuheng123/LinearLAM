import torch
import numpy as np
import matplotlib.pyplot as plt

# save_path_viz = '/home/timpearce/02_lam/01_visualizations_02/' # actions

print('Defaults: Intensity=4.0, action prop=0.0, data_aug=0.0, noise=low epochs=400')

# show effect of noise
path1 = '/home/timpearce/02_lam/01_visualizations_09/save_losses_codes5_intensity0.0_actionprop0.0_k_data_aug0.0.npy'
path2 = '/home/timpearce/02_lam/01_visualizations_09/save_losses_codes5_intensity4.0_actionprop0.0_k_data_aug0.0.npy'
path3 = '/home/timpearce/02_lam/01_visualizations_09/save_losses_codes5_intensity8.0_actionprop0.0_k_data_aug0.0.npy'

data_1 = np.load(path1, allow_pickle=True)
data_2 = np.load(path2, allow_pickle=True)
data_3 = np.load(path3, allow_pickle=True)
print('\n')
print('\t \t \t| Controllable loss \t| Stochastic loss')
print('-'*70)
print('No noise \t \t| ', round(np.mean(data_1[-10:,3])/0.060,3), '\t \t|', round(np.mean(data_1[-10:,4]),3))
print('Low noise \t \t| ', round(np.mean(data_2[-10:,3])/0.060,3), '\t \t|', round(np.mean(data_2[-10:,4])/0.25,3))
print('High noise \t \t| ', round(np.mean(data_3[-10:,3])/0.060,3), '\t \t|', round(np.mean(data_3[-10:,4])/1.0,3))


# 5 codes, intensity 4.0, action prop 0.0, k data aug 0.0
# 0.95 correlation vs 0.0 correlation
path1 = '/home/timpearce/02_lam/01_visualizations_08_cleanrun_lowcommit_dataset0.95/save_losses_codes5_intensity4.0_actionprop0.0_k_data_aug0.0.npy'
path2 = '/home/timpearce/02_lam/01_visualizations_09/save_losses_codes5_intensity4.0_actionprop0.0_k_data_aug0.0.npy'

data_1 = np.load(path1, allow_pickle=True)
data_2 = np.load(path2, allow_pickle=True)
print('\n')
print('\t \t \t| Controllable loss \t| Stochastic loss')
print('-'*70)
print('Uniform policy \t| ', round(np.mean(data_2[-10:,3])/0.060,3), '\t \t|', round(np.mean(data_2[-10:,4])/0.25,3))
print('Correlated policy \t| ', round(np.mean(data_1[-10:,3])/0.060,3), '\t \t|', round(np.mean(data_1[-10:,4])/0.25,3))


# predict action effect
path1 = '/home/timpearce/02_lam/01_visualizations_09/save_losses_codes5_intensity4.0_actionprop0.0_k_data_aug0.0.npy'
path2 = '/home/timpearce/02_lam/01_visualizations_09/save_losses_codes5_intensity4.0_actionprop0.01_k_data_aug0.0.npy'

data_1 = np.load(path1, allow_pickle=True)
data_2 = np.load(path2, allow_pickle=True)
print('\n')
print('\t \t \t| Controllable loss \t| Stochastic loss')
print('-'*70)
print('No action prediction \t| ', round(np.mean(data_1[-10:,3])/0.060,3), '\t \t|', round(np.mean(data_1[-10:,4])/0.25,3))
print('1% action prediction \t| ', round(np.mean(data_2[-10:,3])/0.060,3), '\t \t|', round(np.mean(data_2[-10:,4])/0.25,3))



