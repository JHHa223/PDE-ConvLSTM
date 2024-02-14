import numpy as np
import os

base_path_hsr = '/data0/jhha223/compare_map_plot/mode_anal/2018/'
base_path_prd = '/data0/jhha223/compare_map_plot/mode_anal/2018/'
path_list = os.listdir(base_path_hsr)
path_list = sorted(path_list)

for j in range(len(path_list)):

    #target dataset
    hsr_data_1 = np.load(base_path_hsr+path_list[j]+'/'+path_list[j]+'_HSR_70min.npy')
    hsr_data_2 = np.load(base_path_hsr+path_list[j]+'/'+path_list[j]+'_HSR_80min.npy')
    hsr_data_3 = np.load(base_path_hsr+path_list[j]+'/'+path_list[j]+'_HSR_90min.npy')
    hsr_data_4 = np.load(base_path_hsr+path_list[j]+'/'+path_list[j]+'_HSR_100min.npy')
    hsr_data_5 = np.load(base_path_hsr+path_list[j]+'/'+path_list[j]+'_HSR_110min.npy')
    hsr_data_6 = np.load(base_path_hsr+path_list[j]+'/'+path_list[j]+'_HSR_120min.npy')
    hsr_data_7 = np.load(base_path_hsr+path_list[j]+'/'+path_list[j]+'_HSR_130min.npy')
    hsr_data_8 = np.load(base_path_hsr+path_list[j]+'/'+path_list[j]+'_HSR_140min.npy')
    hsr_data_9 = np.load(base_path_hsr+path_list[j]+'/'+path_list[j]+'_HSR_150min.npy')
    hsr_data_10 = np.load(base_path_hsr+path_list[j]+'/'+path_list[j]+'_HSR_160min.npy')
    hsr_data_11 = np.load(base_path_hsr+path_list[j]+'/'+path_list[j]+'_HSR_170min.npy')
    hsr_data_12 = np.load(base_path_hsr+path_list[j]+'/'+path_list[j]+'_HSR_180min.npy')

    #input data
    ##past radar data    
    prd_data_6m = np.load(base_path_hsr+path_list[j]+'/'+path_list[j]+'_HSR_m50min.npy') 
    prd_data_5m = np.load(base_path_hsr+path_list[j]+'/'+path_list[j]+'_HSR_m40min.npy')
    prd_data_4m = np.load(base_path_hsr+path_list[j]+'/'+path_list[j]+'_HSR_m30min.npy')
    prd_data_3m = np.load(base_path_hsr+path_list[j]+'/'+path_list[j]+'_HSR_m20min.npy')
    prd_data_2m = np.load(base_path_hsr+path_list[j]+'/'+path_list[j]+'_HSR_m10min.npy')
    prd_data_1m = np.load(base_path_hsr+path_list[j]+'/'+path_list[j]+'_HSR_00min.npy')
    ##data produced by advection-diffusion
    prd_data_1 = np.load(base_path_hsr+path_list[j]+'/linear_o/'+path_list[j]+'_HSR_optflow+10min.npy')
    prd_data_2 = np.load(base_path_hsr+path_list[j]+'/linear_o/'+path_list[j]+'_HSR_optflow+20min.npy')
    prd_data_3 = np.load(base_path_hsr+path_list[j]+'/linear_o/'+path_list[j]+'_HSR_optflow+30min.npy')
    prd_data_4 = np.load(base_path_hsr+path_list[j]+'/linear_o/'+path_list[j]+'_HSR_optflow+40min.npy')
    prd_data_5 = np.load(base_path_hsr+path_list[j]+'/linear_o/'+path_list[j]+'_HSR_optflow+50min.npy')
    prd_data_6 = np.load(base_path_hsr+path_list[j]+'/linear_o/'+path_list[j]+'_HSR_optflow+60min.npy')

    #data stacking
    radar_input = np.stack((prd_data_6m,prd_data_5m,prd_data_4m,prd_data_3m,prd_data_2m,prd_data_1m,prd_data_1,prd_data_2,prd_data_3,prd_data_4,prd_data_5,prd_data_6,hsr_data_1,hsr_data_2,hsr_data_3,hsr_data_4,hsr_data_5,hsr_data_6,hsr_data_7,hsr_data_8,hsr_data_9,hsr_data_10,hsr_data_11,hsr_data_12), axis=0)
    radar_input = radar_input.reshape(24,1440,1152,1)

    np.save('/data0/jhha223/conv_LSTM_PDE_tuning/training/input_'+path_list[j]+'.npy',radar_input)
