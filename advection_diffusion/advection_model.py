import cv2
import numpy as np
import math
import glob
import os
import shutil

base_path = 'optflow_vis_test/'
path_list = os.listdir(base_path)
path_list = ['202308291000']

print(path_list)

for ifile in range(len(path_list)):

    if not os.path.isdir('optflow_vis_test/'+path_list[ifile]+'/burger_o'):
        os.makedirs('optflow_vis_test/'+path_list[ifile]+'/burger_o')
    
    new_path = 'optflow_vis_test/'+path_list[ifile]+'/burger_o/'
    
    path = glob.glob(base_path+path_list[ifile]+'/*.npy')
    path = sorted(path)
    print(path)

    shutil.copy(path[0], new_path)
    shutil.copy(path[1], new_path)

    path_tvl1 = glob.glob(base_path+path_list[ifile]+'/burger_o/*.npy')
    path_tvl1 = sorted(path_tvl1)

    for idx in range(0,37):
        #reading input data
        img0 = np.load(path_tvl1[idx])
        img1 = np.load(path_tvl1[idx+1])
    
        #calculating optical flow field
        dtvl1= cv2.optflow.DualTVL1OpticalFlow_create()
        flow_raw = dtvl1.calc(img1, img0, None)
        
        u_raw = flow_raw[:,:,0]
        v_raw = flow_raw[:,:,1]
        R_raw = img1


        #calculating the derivatives
        dx_R = np.zeros((1440,1152))
        dx2_R = np.zeros((1440,1152))
        dy_R = np.zeros((1440,1152))
        dy2_R = np.zeros((1440,1152))

        dx_u = np.zeros((1440,1152))
        dx2_u = np.zeros((1440,1152))
        dy_u = np.zeros((1440,1152))
        dy2_u = np.zeros((1440,1152))

        dx_v = np.zeros((1440,1152))
        dx2_v = np.zeros((1440,1152))
        dy_v = np.zeros((1440,1152))
        dy2_v = np.zeros((1440,1152))

        for i in range(1,1439):
            for j in range(1,1151):
                dx_R[i,j] = (R_raw[i+1,j]-R_raw[i-1,j])/2.
                dx2_R[i,j] = (R_raw[i+1,j]-2*R_raw[i,j]+R_raw[i-1,j])/1.
                dy_R[i,j] = (R_raw[i,j+1]-R_raw[i,j-1])/2.
                dy2_R[i,j] = (R_raw[i,j+1]-2*R_raw[i,j]+R_raw[i,j-1])/1.

                dx_u[i,j] = (u_raw[i+1,j]-u_raw[i-1,j])/2.
                dx2_u[i,j] = (u_raw[i+1,j]-2*u_raw[i,j]+u_raw[i-1,j])/1.
                dy_u[i,j] = (u_raw[i,j+1]-u_raw[i,j-1])/2.
                dy2_u[i,j] = (u_raw[i,j+1]-2*u_raw[i,j]+u_raw[i,j-1])/1.

                dx_v[i,j] = (v_raw[i+1,j]-v_raw[i-1,j])/2.
                dx2_v[i,j] = (v_raw[i+1,j]-2*v_raw[i,j]+v_raw[i-1,j])/1.
                dy_v[i,j] = (v_raw[i,j+1]-v_raw[i,j-1])/2.
                dy2_v[i,j] = (v_raw[i,j+1]-2*v_raw[i,j]+v_raw[i,j-1])/1.

        #kinematic viscosity and diffusion coefficient
        coeff_mu = 0.2
        coeff_nu = 0.05

        #Updating flow field
        flow_raw[:,:,0] = u_raw - u_raw*dx_u-v_raw*dy_u + coeff_mu*(dx2_u+dy2_u)
        flow_raw[:,:,1] = v_raw - u_raw*dx_v-v_raw*dy_v + coeff_mu*(dx2_v+dy2_v)
        
        #Updating precipitation field: advection
        flow_new = flow_raw
        h,w,_ = flow_new.shape
        flow_new[:,:,0] += np.arange(w)
        flow_new[:,:,1] += np.arange(h)[:,np.newaxis]
        R_adv = cv2.remap(img1, flow_new, None, cv2.INTER_CUBIC)

        #Updating precipitation field: diffusion
        R_adv = R_adv + coeff_nu*(dx2_R+dy2_R)
        R_adv = R_adv.astype(np.float32)

        #Saving the updated precipitation field
        name = new_path+'o_p_burger_'+path_list[ifile]+'+'+str(10*(idx+1))+'min.npy'
        np.save(new_path+'o_p_burger_'+path_list[ifile]+'+'+str(10*(idx+1))+'min.npy',R_adv)
        path_tvl1.append(name)
