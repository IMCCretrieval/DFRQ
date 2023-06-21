from semi_config import *
from PIL import Image, ImageEnhance, ImageOps, ImageFile
# from balance_triplet_train import *

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_data_one(file):
    batch = unpickle(file)
    data = batch[b'data']
    labels = batch[b'labels']
    #print("Loading %s : %d." % (file, len(data)))
    return data, labels


def load_data(files, data_dir, label_count):
    global image_size, img_channels
    data, labels = load_data_one(data_dir + '/' + files[0])
    for f in files[1:]:
        data_n, labels_n = load_data_one(data_dir + '/' + f)
        data = np.append(data, data_n, axis=0)
        labels = np.append(labels, labels_n, axis=0)
    labels = np.array([[float(i == label) for i in range(label_count)] for label in labels])
    data = data.reshape([-1, img_channels, image_size, image_size])
    data = data.transpose([0, 2, 3, 1])
    return data, labels


def label_Sim_Generate(traingnd, testgnd):

    numtrain = traingnd.shape[0]
    numtest = testgnd.shape[0]

    label_Sim = np.zeros((numtest, numtrain))

    for i in range(numtest):
        label_Sim[i,:] = (testgnd[i]==traingnd).astype(int)
        if i%1000 == 0:
            print("%d-th processing"%(i))

    return label_Sim
def color_preprocessing(x_):

    #Normalize with mean and std of source data
    x_ = x_.astype('float32')

    x_[:, :, :, 0] = (x_[:, :, :, 0] - 125.642) / 63.01
    x_[:, :, :, 1] = (x_[:, :, :, 1] - 123.738) / 62.157
    x_[:, :, :, 2] = (x_[:, :, :, 2] - 114.46) / 66.94

    return x_
def prepare_data(data_dir, train_flag):
    print("======Loading data======")
    meta = unpickle(data_dir + '/batches.meta')

    label_names = meta[b'label_names']
    label_count = len(label_names)
    train_files = ['data_batch_%d' % d for d in range(1, 6)]
    train_x, train_y = load_data(train_files, data_dir, label_count)
    test_x, test_y = load_data(['test_batch'], data_dir, label_count)

    train_x = train_x.astype(float)
    test_x = test_x.astype(float)
    train_y = np.argmax(train_y, axis=1).astype(int)
    test_y = np.argmax(test_y, axis=1).astype(int)

    Total_x = np.concatenate((train_x, test_x), axis=0)
    
    Total_x = color_preprocessing(Total_x)
    
    Total_y = np.concatenate((train_y, test_y), axis=0)

    argidx = np.argsort(Total_y)

    Total_x = Total_x[argidx]
    Total_y = Total_y[argidx]
    # start_num=0 #1能够达到0.881
    # num_select=60
    # Query_x = Total_x[start_num::num_select]# step=60 60000/60==1000 从0开始
    # Query_y = Total_y[start_num::num_select]

    # Gallery_x = np.delete(Total_x, np.arange(start_num, np.shape(Total_x)[0], num_select), axis=0)
    # Gallery_y = np.delete(Total_y, np.arange(start_num, np.shape(Total_y)[0], num_select), axis=0)

    Gallery_split_x = np.split(Total_x, n_CLASSES, 0)
    Gallery_split_y = np.split(Total_y, n_CLASSES, 0)

    for i in range(n_CLASSES):
        if i<7:
            train75_tmp_x = Gallery_split_x[i][:3000]
            train75_tmp_y = Gallery_split_y[i][:3000]
            test75_tmp_x = Gallery_split_x[i][3000:]
            test75_tmp_y = Gallery_split_y[i][3000:]
            if i == 0:
                train75_x = train75_tmp_x
                train75_y = train75_tmp_y
                test75_x = test75_tmp_x
                test75_y = test75_tmp_y
            else:
                train75_x = np.concatenate((train75_x, train75_tmp_x), axis=0)
                train75_y = np.concatenate((train75_y, train75_tmp_y), axis=0)
                test75_x = np.concatenate((test75_x, test75_tmp_x), axis=0)
                test75_y = np.concatenate((test75_y, test75_tmp_y), axis=0)
        else:
            train25_tmp_x = Gallery_split_x[i][:3000]
            train25_tmp_y = Gallery_split_y[i][:3000]
            test25_tmp_x = Gallery_split_x[i][3000:]
            test25_tmp_y = Gallery_split_y[i][3000:]
            if i == 7:
                train25_x = train25_tmp_x
                train25_y = train25_tmp_y
                test25_x = test25_tmp_x
                test25_y = test25_tmp_y
            else:
                train25_x = np.concatenate((train25_x, train25_tmp_x), axis=0)
                train25_y = np.concatenate((train25_y, train25_tmp_y), axis=0)
                test25_x = np.concatenate((test25_x, test25_tmp_x), axis=0)
                test25_y = np.concatenate((test25_y, test25_tmp_y), axis=0)

    Gallery_x = np.concatenate((train25_x, test75_x), axis=0)
    Gallery_y = np.concatenate((train25_y, test75_y), axis=0)

    Source_x = train75_x
    Source_y = train75_y
    Target_x = Gallery_x
    Query_x = test25_x
    Query_y = test25_y
    if train_flag == True:
        print("Source: ", np.shape(Source_x), np.shape(Source_y))
        print("Target: ", np.shape(Target_x))
        print("Gallery: ", np.shape(Gallery_x), np.shape(Gallery_y))
        print("Query: ", np.shape(Query_x), np.shape(Query_y))
        #del Gallery_x, Gallery_y, Query_x, Query_y
        data_config = Source_x, Source_y, Target_x, Gallery_x, Gallery_y, Query_x, Query_y
    '''
    else:
        print("Gallery: ", np.shape(Gallery_x), np.shape(Gallery_y))
        print("Query: ", np.shape(Query_x), np.shape(Query_y))
        del Source_x, Source_y, Target_x
        data_config = Gallery_x, Gallery_y, Query_x, Query_y
    del Total_x, Total_y
    ''' 
    return data_config
'''  
def prepare_data(data_dir, train_flag):
    print("======Loading data======")
    meta = unpickle(data_dir + '/batches.meta')

    label_names = meta[b'label_names']
    label_count = len(label_names)
    train_files = ['data_batch_%d' % d for d in range(1, 6)]
    train_x, train_y = load_data(train_files, data_dir, label_count)
    test_x, test_y = load_data(['test_batch'], data_dir, label_count)

    train_x = train_x.astype(float)
    test_x = test_x.astype(float)
    train_y = np.argmax(train_y, axis=1).astype(int)
    test_y = np.argmax(test_y, axis=1).astype(int)

    Total_x = np.concatenate((train_x, test_x), axis=0)
    
    Total_x = color_preprocessing(Total_x)
    
    Total_y = np.concatenate((train_y, test_y), axis=0)

    argidx = np.argsort(Total_y)

    Total_x = Total_x[argidx]
    Total_y = Total_y[argidx]

    Query_x = Total_x[::60]# step=60 60000/60==1000
    Query_y = Total_y[::60]

    Gallery_x = np.delete(Total_x, np.arange(0, np.shape(Total_x)[0], 60), axis=0)
    Gallery_y = np.delete(Total_y, np.arange(0, np.shape(Total_y)[0], 60), axis=0)

    Gallery_split_x = np.split(Gallery_x, n_CLASSES, 0)
    Gallery_split_y = np.split(Gallery_y, n_CLASSES, 0)

    for i in range(n_CLASSES):
        Source_tmp_x = Gallery_split_x[i][:500]
        Source_tmp_y = Gallery_split_y[i][:500]
        Target_tmp_x = Gallery_split_x[i][500:]
        Target_tmp_y = Gallery_split_y[i][500:]

        if i == 0:
            Source_x = Source_tmp_x
            Source_y = Source_tmp_y
            Target_x = Target_tmp_x
            Target_y = Target_tmp_y
        else:
            Source_x = np.concatenate((Source_x, Source_tmp_x), axis=0)
            Source_y = np.concatenate((Source_y, Source_tmp_y), axis=0)
            Target_x = np.concatenate((Target_x, Target_tmp_x), axis=0)
            Target_y = np.concatenate((Target_y, Target_tmp_y), axis=0)

    Gallery_x = Target_x
    Gallery_y = Target_y

    if train_flag == True:
        print("Source: ", np.shape(Source_x), np.shape(Source_y))
        print("Target: ", np.shape(Target_x))
        del Gallery_x, Gallery_y, Query_x, Query_y
        data_config = Source_x, Source_y, Target_x
    else:
        print("Gallery: ", np.shape(Gallery_x), np.shape(Gallery_y))
        print("Query: ", np.shape(Query_x), np.shape(Query_y))
        del Source_x, Source_y, Target_x
        data_config = Gallery_x, Query_x
    del Total_x, Total_y

    return data_config
'''
if __name__ == '__main__':
    prepare_data(data_dir, True)
