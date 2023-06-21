from nus_config import *

from PIL import Image
# from skimage import io, transform

import time
# import cv2
import scipy.io as io
import h5py
def color_preprocessing(x_):
#会不会这里出问题了
    #Normalize with mean and std of nuswide training  
    x_ = x_.astype('float32')

#     x_[:, :, :, 0] = (x_[:, :, :, 0] - 111.804) / 71.349 # np.mean(train_x[:,:,:,0]) np.std
#     x_[:, :, :, 1] = (x_[:, :, :, 1] - 107.83) / 67.951
#     x_[:, :, :, 2] = (x_[:, :, :, 2] - 99.814) / 73.239
    
    #cifar
    x_[:, :, :, 0] = (x_[:, :, :, 0] - 125.642) / 63.01
    x_[:, :, :, 1] = (x_[:, :, :, 1] - 123.738) / 62.157
    x_[:, :, :, 2] = (x_[:, :, :, 2] - 114.46) / 66.94
    

    return x_

def label_Sim_Generate(traingnd, testgnd):

    numtrain = traingnd.shape[0]
    numtest = testgnd.shape[0]

    label_Sim = np.zeros((numtest, numtrain))

    for i in range(numtest):
        for j in range(numtrain):
            if np.dot(traingnd[j],testgnd[i])!=0:
                label_Sim[i,j] = 1
        if i%100 == 0:
            print("%d-th processing"%(i))

    return label_Sim


def load_nus_data(files,label_dir, data_dir):
    global image_size, img_channels
    print(image_size)
    img_filepath = os.path.join(data_dir, files)
    label_filepath = os.path.join(data_dir, label_dir)
    fp = open(img_filepath, 'r')
    img_filename = [x.strip() for x in fp]
    fp.close()
    
    labels = np.loadtxt(label_filepath, dtype=np.int64)
   
    time_start=time.time()
    
    data= np.zeros([labels.shape[0],image_size,image_size,3], dtype = float)
    count=0
#     data = np.array(Image.open(os.path.join(data_dir, img_filename[0])).resize((image_size,image_size)))
    for f in img_filename :
        data_n = np.array(Image.open(os.path.join(data_dir, f)).resize((image_size,image_size)))
#         data = np.append(data, data_n, axis=0)#越来越慢！！
        data[count]=data_n

        count=count+1

    print(count)
    time_end=time.time() #508s
    print('PIL time cost',time_end-time_start,'s')   
    return data, labels

def prepare_data(data_dir, train_dir,train_label,test_dir,test_label,database_dir,db_label,train_flag):
    print("======Loading data======")
    if train_flag == True:
        train_x, train_y = load_nus_data(train_dir,train_label, data_dir)
        
        train_x = color_preprocessing(train_x)
        print("Source: ", np.shape(train_x), np.shape(train_y))

        data_config = train_x, train_y
    else:
        database_x, database_y = load_nus_data(database_dir,db_label, data_dir)
        
        database_x = color_preprocessing(database_x)
        test_x, test_y = load_nus_data(test_dir,test_label, data_dir)  
        test_x = color_preprocessing(test_x)
        print("Gallery: ", np.shape(database_x), np.shape(database_y))
        print("Query: ", np.shape(test_x), np.shape(test_y))
        data_config = database_x,database_y, test_x,test_y

    return data_config

if __name__ == "__main__":

    Source_x, Source_y = prepare_data(data_dir, train_dir,train_label,test_dir,test_label,database_dir,database_label, True)
    f_train = h5py.File('/gdata2/lipd/NUS-WIDE/train_x_64_data.h5', 'w')
    f_train.create_dataset('train_x', data=Source_x)
    f_train.close()
    
    ff_train1 = h5py.File('/gdata2/lipd/NUS-WIDE/train_y_64_data.h5', 'w')
    ff_train1.create_dataset('train_y', data=Source_y)
    ff_train1.close()
    
    db_x, db_y, q_x, q_y  = prepare_data(data_dir, train_dir,train_label,test_dir,test_label,database_dir,database_label, False)
    

    nus_sim = label_Sim_Generate(db_y,q_y)
    f2 = h5py.File('/gdata2/lipd/NUS-WIDE/nus_sim.h5', 'w')
    f2.create_dataset('nus_sim', data=nus_sim)
    f2.close()
    
    f = h5py.File('/gdata2/lipd/NUS-WIDE/database_nus_64_data.h5', 'w')
    f.create_dataset('database_x', data=db_x)
    f.close()
    
    f1 = h5py.File('/gdata2/lipd/NUS-WIDE/query_nus_64_data.h5', 'w')
    f1.create_dataset('query_x', data=q_x)
    f1.close()

# if __name__ == "__main__":
# #     Source_x, Source_y = prepare_data(data_dir, train_dir,train_label,test_dir,test_label,database_dir,database_label, True)
#     _, db_y, _, q_y  = prepare_data(data_dir, train_dir,train_label,test_dir,test_label,database_dir,database_label, False)
#     nus_sim = label_Sim_Generate(db_y,q_y)
#     mat_path = './nus_sim.mat'
#     io.savemat(mat_path, {'name': nus_sim})