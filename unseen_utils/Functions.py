
import random
from semi_config import *
from unseen_utils.cifar_unseen import *
# from balance_triplet_train import *
from PIL import Image, ImageEnhance, ImageOps, ImageFile, ImageFilter
from tensorflow.contrib.framework import arg_scope
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, xavier_initializer
import sys
import math

import numpy as np
 
import time
initializer = xavier_initializer()


def _fft_image(img):
    f1 = np.fft.fft2(img)
    fshift1 = np.fft.fftshift(f1)
    pha=np.angle(fshift1)
    amp=np.abs(fshift1)
    return amp, pha

def _new_ifft_img(amp,pha):
    new_fshift=amp*np.exp(1j*pha)
    iamp=np.fft.ifftshift(new_fshift)
    tmp=np.fft.ifft2(iamp)
    return np.abs(tmp)

#20210426
def _ssl_fft_beta(data,rank_num):
    data_num=data.shape[0]
    h=data.shape[2]
    w=data.shape[3]

    # r=0.15# r=0.2 效果0.885 r=0.1,map0.8851
    r=0.1*(rank_num)
    r_h,r_w =np.ceil(h*r/2), np.ceil(w*r/2)
    x1,y1=int(h/2-r_h), int(w/2-r_w)
    x2,y2=int(h/2+r_h), int(w/2+r_w)

    data_split=np.copy(data)
    indices = np.random.permutation(data_num) 
    amp=np.zeros_like(data)
    pha=np.zeros_like(data)
    #fft
    for j in range(3):
        for i in range(data_num):
            amp[i,:,:,j],pha[i,:,:,j] = _fft_image(data_split[i,:,:,j])

        #swap
        #amp[:,x1:x2,y1:y2,j]=amp[indices,x1:x2,y1:y2,j]
        pha[:,x1:x2,y1:y2,j]=(1.0-rank_num*0.2)*pha[:,x1:x2,y1:y2,j]+(rank_num*0.2)*pha[indices,x1:x2,y1:y2,j] #beta
        # pha[:,x1:x2,y1:y2,j]=(1.0-rank_num*0.3)*pha[:,x1:x2,y1:y2,j]+(rank_num*0.3)*pha[indices,x1:x2,y1:y2,j] #cuda3
        
    #ifft
    for j in range(3):
        for i in range(data_num):
            data_split[i,:,:,j]=_new_ifft_img(amp[i,:,:,j],pha[i,:,:,j])
    return data_split

#20210426
def _ssl_fft(data,rank_num):
    data_num=data.shape[0]
    h=data.shape[2]
    w=data.shape[3]

    r=0.02+(rank_num)*0.07 #r=(rank_num+1)*0.04 效果达到0.876
    
    r_h,r_w =np.ceil(h*r/2), np.ceil(w*r/2)
    x1,y1=int(h/2-r_h), int(w/2-r_w)
    x2,y2=int(h/2+r_h), int(w/2+r_w)

    data_split=np.copy(data)
    indices = np.random.permutation(data_num) 
    amp=np.zeros_like(data)
    pha=np.zeros_like(data)
    #fft
    for j in range(3):
        for i in range(data_num):
            amp[i,:,:,j],pha[i,:,:,j] = _fft_image(data_split[i,:,:,j])

        #swap
        #amp[:,x1:x2,y1:y2,j]=amp[indices,x1:x2,y1:y2,j]
        pha[:,x1:x2,y1:y2,j]=pha[indices,x1:x2,y1:y2,j] #交换相位
        
    #ifft
    for j in range(3):
        for i in range(data_num):
            data_split[i,:,:,j]=_new_ifft_img(amp[i,:,:,j],pha[i,:,:,j])
    return data_split



## fft rank augmentation
def unnlabel_fft_augmentation(batch, crop_size=32,count=4):
    all_batch= _random_crop(batch, [crop_size, crop_size], 4)
    for i in range(count):
        #tmp=_ssl_fft(batch,i)
        tmp=_ssl_fft_beta(batch,i)
        all_batch=np.concatenate((all_batch, _random_crop(tmp, [crop_size, crop_size], 4*i)),axis=0) ##2 or 4
    return all_batch

## fft+cut augmentation
def unnlabel_fft_cut_augmentation(batch, crop_size=32,count=4):
    all_batch= _random_crop(batch, [crop_size, crop_size], 4)
    for i in range(count):
        tmp=_ssl_cutmix(batch,i,scale=0.08)
        tmp=_ssl_fft(tmp,i)
        all_batch=np.concatenate((all_batch, _random_crop(tmp, [crop_size, crop_size], 4*i)),axis=0) ##2 or 4
    return all_batch

'''
scale尺度因子，控制loss大小，之前默认scale=3，scale为9，loss小，性能下降0.01，scale为2，loss大.
sim_beta默认0.5
sim_margin默认0.03，0.05效果不好
'''
def self_ranking_loss(features_x,features_q, batch_size,count=4,scale=3,sim_margin=0.03,sim_beta=0.5):
    features_x = tf.nn.l2_normalize(features_x, axis=1)
    features_q = tf.nn.l2_normalize(features_q, axis=1)   
    sim = tf.stack([tf.einsum('nc,nc->n', features_x[0:batch_size], features_q[batch_size*i:batch_size*(i+1)]) for i in range(1, count)], axis=0)

    rank_high = sim[:-1]
    rank_low = sim[1:]
    
    loss_sort = tf.reduce_mean(tf.log(1 + tf.reduce_sum(tf.exp( scale * (-rank_high + rank_low + sim_margin)), axis=0)))

    loss_pos = tf.reduce_mean(tf.log(1 + tf.reduce_sum(tf.exp(- scale * (sim - sim_beta)), axis=0)))
    
    loss = (loss_sort + loss_pos) /  scale
    # loss = loss_sort /  scale           
    return loss

#20210426
def _ssl_cutmix(data,rank_num, scale=0.08):
    data_num=data.shape[0]
    data_split=data
    indices = np.random.permutation(data_num) 
    lam = 0.02+scale*(rank_num+1)
#         lam = 0.02+0.15*(i+1)
#         lam = 0.02+0.05*(i+1)
#         lam = 0.02+0.12*(i+1)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.shape, lam)
    data_split[:, bbx1:bbx2, bby1:bby2,:] = data[indices, bbx1:bbx2, bby1:bby2, :]
    return data_split


## rank augmentation
def unnlabel_data_augmentation(batch, crop_size=32,count=4):
    all_batch= _random_crop(batch, [crop_size, crop_size], 4)
    for i in range(count):
        tmp=_ssl_cutmix(batch,i)
        all_batch=np.concatenate((all_batch, _random_crop(tmp, [crop_size, crop_size], 4*i)),axis=0) 
    return all_batch



def rand_bbox(size, lam):
    W = size[1]
    H = size[2]
    cut_rat =lam
#     cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def remove_diag(M):
#     h, w = M.shape
#     print(M.shape)
#     assert h==w, "h and w should be same"
    mask = np.ones((batch_size, batch_size)) - np.eye(batch_size)

    mask = (mask).astype(bool) 
    return tf.reshape(M[mask],[batch_size, -1])
def Multi_Contrastive_quantization_loss(labels_Similarity, embeddings_x, embeddings_q,  n_book, margin=0.0, temperature=1.0):    
    reg_anchor = tf.reduce_mean(tf.reduce_sum(tf.square(embeddings_x), 1))
    reg_positive = tf.reduce_mean(tf.reduce_sum(tf.square(embeddings_q), 1))
    l2loss = tf.multiply(0.25 * 0.002, reg_anchor + reg_positive, name='l2loss')

    embeddings_x = tf.nn.l2_normalize(embeddings_x, axis=1)
    embeddings_q = tf.nn.l2_normalize(embeddings_q, axis=1)

    temperature=1/n_book
    print(temperature)
    similarity_matrix= tf.matmul(embeddings_x, embeddings_q, transpose_a=False, transpose_b=True) 
    # similarity_matrix= tf.matmul(embeddings_x, embeddings_x, transpose_a=False, transpose_b=True) 
    instance_zone = tf.exp((remove_diag(similarity_matrix) - margin)/temperature)#remove diag element 
    
    inst2proxy_positive =tf.exp((tf.matmul(embeddings_x, embeddings_q, transpose_a=False, transpose_b=True) - margin)/temperature)#视觉向量和嵌入的计算
    
    positive_samples=tf.multiply(instance_zone,remove_diag(labels_Similarity)) ##500,499  
    positive_score =tf.reduce_sum(positive_samples,1)#500,1 分子求和项
    
#     negative_samples=tf.multiply(instance_zone,1-remove_diag(labels_Similarity)) ##500,499
    negative_samples=instance_zone
    negative_score =tf.reduce_sum(negative_samples,1)#500,1  分母求和项
    
    x_q=tf.diag_part(inst2proxy_positive) # 此视觉向量和嵌入的内积在对角线上， 
    
    numerator=x_q+positive_score#分子
    denomerator=x_q+negative_score#分母
    
    loss=tf.reduce_mean(-tf.log(numerator/denomerator))
    
    return loss +l2loss


def MS_loss(labels, embeddings_x, embeddings_q, al =2.0, be =50.0, lamb=1.0, eps=0.1, ms_mining=True):
    '''
    ref: http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf
    official codes: https://github.com/MalongTech/research-ms-loss
    '''
#     print(labels_Similarity)

    # make sure emebedding should be l2-normalized
    #embeddings_x = tf.nn.l2_normalize(embeddings_x, axis=1)
    #embeddings_q = tf.nn.l2_normalize(embeddings_q, axis=1)


#     batch_size = embeddings_x.get_shape().as_list()[0]

    labels = tf.reshape(labels, [-1, 1]) #张量变为一维列向量
    adjacency = tf.equal(labels, tf.transpose(labels))
#     adjacency = labels_Similarity
    adjacency_not = tf.logical_not(adjacency)
    
    
    mask_neg = tf.cast(adjacency_not, dtype=tf.float32)
    mask_pos = tf.cast(adjacency, dtype=tf.float32) - tf.eye(batch_size, dtype=tf.float32)
  

    sim_mat = tf.matmul(embeddings_x, embeddings_q, transpose_a=False, transpose_b=True)
    sim_mat = tf.maximum(sim_mat, 0.0)

    pos_mat = tf.multiply(sim_mat, mask_pos)
    neg_mat = tf.multiply(sim_mat, mask_neg)

    if ms_mining:#会降低很多，为什么
        max_val = tf.reduce_max(neg_mat, axis=1, keepdims=True)
        tmp_max_val = tf.reduce_max(pos_mat, axis=1, keepdims=True)
        min_val = tf.reduce_min(tf.multiply(sim_mat - tmp_max_val, mask_pos), axis=1, keepdims=True) + tmp_max_val

        max_val = tf.tile(max_val, [1, batch_size])
        min_val = tf.tile(min_val, [1, batch_size])

        mask_pos = tf.where(pos_mat < max_val + eps, mask_pos, tf.zeros_like(mask_pos))
        mask_neg = tf.where(neg_mat > min_val - eps, mask_neg, tf.zeros_like(mask_neg))

    pos_exp = tf.exp(-al  * (pos_mat - lamb))
    pos_exp = tf.where(mask_pos > 0.0, pos_exp, tf.zeros_like(pos_exp))

    neg_exp = tf.exp(be * (neg_mat - lamb))
    neg_exp = tf.where(mask_neg > 0.0, neg_exp, tf.zeros_like(neg_exp))

    pos_term = tf.log(1.0 + tf.reduce_sum(pos_exp, axis=1)) / al 
    neg_term = tf.log(1.0 + tf.reduce_sum(neg_exp, axis=1)) / be 

    loss = tf.reduce_mean(pos_term + neg_term)

    return loss


# def random_flip(flip_x_chance, flip_y_chance, prng=DEFAULT_PRNG):
#     """ Construct a transformation randomly containing X/Y flips (or not).
#     Args
#         flip_x_chance: The chance that the result will contain a flip along the X axis.
#         flip_y_chance: The chance that the result will contain a flip along the Y axis.
#         prng:          The pseudo-random number generator to use.
#     Returns
#         a homogeneous 3 by 3 transformation matrix
#     """
#     flip_x = prng.uniform(0, 1) < flip_x_chance
#     flip_y = prng.uniform(0, 1) < flip_y_chance
#     # 1 - 2 * bool gives 1 for False and -1 for True.
#     return scaling((1 - 2 * flip_x, 1 - 2 * flip_y))



# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                                  std=[0.229, 0.224, 0.225])   

#20210122 1740 crop 0.15->0.1

def data_augmentation(batch, crop_size=32):
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [crop_size, crop_size], 4)
    return batch

def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                       nw:nw + crop_shape[1]]
    return new_batch

def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv_layer(input, filter, kernel, stride=1, layer_name="conv", padding='SAME'):
    with tf.name_scope(layer_name):
        conv = tf.layers.conv2d(inputs=input, use_bias=True, filters=filter, kernel_size=kernel, strides=stride, padding=padding, kernel_initializer=initializer)
        return  conv

def Linear(x, out_length, layer_name) :
    with tf.name_scope(layer_name):
        linear = tf.layers.dense(inputs=x, units=out_length, kernel_initializer=initializer)
        return linear

def Batch_Normalization(x, training, scope="batch"):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def Relu(x):
    return tf.nn.relu(x)

def SoftMax(x, axis=-1):
    return tf.nn.softmax(x, axis=axis)

def Max_Pooling(x, pool_size=[2,2], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Global_Average_Pooling(x):
     return global_avg_pool(x, name='Global_avg_pooling')

def flip_gradient(x, l=1.0):
	positive_path = tf.stop_gradient(x * tf.cast(1 + l, tf.float32))
	negative_path = -x * tf.cast(l, tf.float32)
	return positive_path + negative_path


def Soft_Assignment(z_, x_, n_book, alpha):
    x = tf.split(x_, n_book, 1)
    y = tf.split(z_, n_book, 1)
    for i in range(n_book):
        size_x = tf.shape(x[i])[0] #batch? 
        size_y = tf.shape(y[i])[0]
        xx = tf.expand_dims(x[i], -1)
        xx = tf.tile(xx, tf.stack([1, 1, size_y])) #copy   times in third-d

        yy = tf.expand_dims(y[i], -1)
        yy = tf.tile(yy, tf.stack([1, 1, size_x]))
        yy = tf.transpose(yy, perm=[2, 1, 0])
        diff = 1-tf.reduce_sum(tf.multiply(xx,yy), 1)# element product...confuse about 1-
        softmax_diff = SoftMax(diff * (-alpha), 1)
        # print(tf.reduce_max(softmax_diff,1))
        if i==0:
            # index_max=tf.argmax(softmax_diff,1)

            soft_des_tmp = tf.matmul(softmax_diff, y[i], transpose_a=False, transpose_b=False)
            descriptor = soft_des_tmp
            # tmp_index = index_max
            tmp_diff=softmax_diff
        else:
            # index_max=tf.argmax(softmax_diff,1)
            
            soft_des_tmp = tf.matmul(softmax_diff, y[i], transpose_a=False, transpose_b=False)
            descriptor = tf.concat([descriptor, soft_des_tmp], axis=1)
            
            # tmp_index =tf.concat([tmp_index, index_max], axis=0)
            tmp_diff =tf.concat([tmp_diff, softmax_diff], axis=1)

    return Intra_Norm(descriptor, n_book), tmp_diff

'''
import operator
def myED(testdata,traindata,batch_size):
    """ 计算欧式距离，要求测试样本和训练样本以array([ [],[],...[] ])的形式组织，
    每行表示一个样本，一列表示一个属性"""
    # size_train=traindata.shape[0] # 训练样本量大小
    # size_test=testdata.shape[0] # 测试样本大小
    b_list = []

    for i in range (batch_size):
        distance=tf.reduce_sum(tf.abs(tf.add(traindata,tf.negative(testdata[i,:]))),axis=1)
        nn_index=tf.argmin(distance,axis=0)
        # testdata[i,:]=traindata[nn_index,:]
        b_list.append(traindata[nn_index,:])
    
    aaa = tf.stack(b_list)
# distance=tf.reduce_sum(tf.abs(tf.add(xtr,tf.negative(xte))),axis=1)
# nn_index=tf.argmin(distance,axis=0)

    # XX=traindata**2
    # sumXX=tf.reduce_sum(XX, axis=1) # 行平方和
    # YY=testdata**2
    # sumYY=tf.reduce_sum(YY, axis=1) # 行平方和
    # Xpw2_plus_Ypw2=tf.tile(mat(sumXX).T,[1,size_test]) + tf.tile(mat(sumYY),[size_train,1])
    # EDsq=Xpw2_plus_Ypw2-2*(mat(traindata)*mat(testdata).T) # 欧式距离平方
    # distances=EDsq**0.5 #欧式距离
    return aaa
def nnclr_rm_diag(M):
#     h, w = M.shape
#     print(M.shape)
#     assert h==w, "h and w should be same"
    mask = np.ones((unlabel_batch_size, unlabel_batch_size)) - np.eye(unlabel_batch_size)

    mask = (mask).astype(bool) 
    return tf.reshape(M[mask],[unlabel_batch_size, -1])
def nnclr(features_x,features_q,memory_bank,batch_size,margin=0.0,temperature=1.0):
    features_x = tf.nn.l2_normalize(features_x, axis=1)
    features_q = tf.nn.l2_normalize(features_q, axis=1)
    memory_bank= tf.nn.l2_normalize(memory_bank, axis=1)
    features_tmp=features_x  
    features_tmp=myED(features_tmp,memory_bank,batch_size)
    
    similarity_matrix= tf.matmul(features_tmp, features_q, transpose_a=False, transpose_b=True) 
    # similarity_matrix= tf.matmul(embeddings_x, embeddings_x, transpose_a=False, transpose_b=True) 
    instance_zone = tf.exp((nnclr_rm_diag(similarity_matrix) - margin)/temperature)#remove diag element 
    inst2proxy_positive =tf.exp((tf.matmul(features_tmp, features_q, transpose_a=False, transpose_b=True) - margin)/temperature)#视觉向量和嵌入的计算
    negative_samples=instance_zone
    negative_score =tf.reduce_sum(negative_samples,1)#500,1  分母求和项
    x_q=tf.diag_part(inst2proxy_positive) # 此视觉向量和嵌入的内积在对角线上， 
    
    numerator=x_q #分子
    denomerator=x_q+negative_score#分母
    
    loss=tf.reduce_mean(-tf.log(numerator/denomerator))
    return loss

def self_nn_ranking(features_x,features_q,memory_bank, batch_size,count=3,scale=3,sim_margin=0.03,sim_beta=0.5):
    features_x = tf.nn.l2_normalize(features_x, axis=1)
    features_q = tf.nn.l2_normalize(features_q, axis=1)
    memory_bank= tf.nn.l2_normalize(memory_bank, axis=1)
    features_tmp=features_x  
    features_tmp=myED(features_tmp,memory_bank,batch_size)
    # Dsortindex=Dis.argsort(axis=0) # 距离排序，提取序号
    # nearest_k=Dsortindex[0,:] # 提取最近k个距离的样本序号
    sim = tf.stack([tf.einsum('nc,nc->n', features_tmp[0:batch_size], features_q[batch_size*i:batch_size*(i+1)]) for i in range(1, count)], axis=0)

    rank_high = sim[:-1]
    rank_low = sim[1:]
    
    loss_sort = tf.reduce_mean(tf.log(1 + tf.reduce_sum(tf.exp( scale * (-rank_high + rank_low + sim_margin)), axis=0)))

    loss_pos = tf.reduce_mean(tf.log(1 + tf.reduce_sum(tf.exp(- scale * (sim - sim_beta)), axis=0)))
    
    loss = (loss_sort + loss_pos) /  scale
    # loss = loss_sort /  scale           
    return loss
'''
# def Soft_Assignment(z_, x_, n_book, alpha):

#     x = tf.split(x_, n_book, 1)
#     y = tf.split(z_, n_book, 1)
#     for i in range(n_book):
#         size_x = tf.shape(x[i])[0]
#         size_y = tf.shape(y[i])[0]
#         xx = tf.expand_dims(x[i], -1)
#         xx = tf.tile(xx, tf.stack([1, 1, size_y]))

#         yy = tf.expand_dims(y[i], -1)
#         yy = tf.tile(yy, tf.stack([1, 1, size_x]))
#         yy = tf.transpose(yy, perm=[2, 1, 0])

#         diff = 1-tf.reduce_sum(tf.multiply(xx,yy), 1)
#         softmax_diff = SoftMax(diff * (-alpha), 1)

#         if i==0:
#             soft_des_tmp = tf.matmul(softmax_diff, y[i], transpose_a=False, transpose_b=False)
#             descriptor = soft_des_tmp
#         else:
#             soft_des_tmp = tf.matmul(softmax_diff, y[i], transpose_a=False, transpose_b=False)
#             descriptor = tf.concat([descriptor, soft_des_tmp], axis=1)

#     return Intra_Norm(descriptor, n_book)

def Intra_Norm(features, numSeg):
    x = tf.split(features, numSeg, 1)
    for i in range(numSeg):
        norm_tmp = tf.nn.l2_normalize(x[i], axis=1)
        if i==0:
            innorm = norm_tmp
        else:
            innorm = tf.concat([innorm, norm_tmp], axis=1)
    return innorm

# N_pair Product Quantization loss
def N_PQ_loss(labels_Similarity, embeddings_x, embeddings_q, n_book, reg_lambda=0.002):

  reg_anchor = tf.reduce_mean(tf.reduce_sum(tf.square(embeddings_x), 1))
  reg_positive = tf.reduce_mean(tf.reduce_sum(tf.square(embeddings_q), 1))
  l2loss = tf.multiply(0.25 * reg_lambda, reg_anchor + reg_positive, name='l2loss')
  
  embeddings_x = tf.nn.l2_normalize(embeddings_x, axis=1)
  embeddings_q = tf.nn.l2_normalize(embeddings_q, axis=1)

  FQ_Similarity = tf.matmul(embeddings_x, embeddings_q, transpose_a=False, transpose_b=True)*n_book

  # Add the softmax loss.
  loss = tf.nn.softmax_cross_entropy_with_logits(logits=FQ_Similarity, labels=labels_Similarity)
  loss = tf.reduce_mean(loss, name='xentropy')

  return l2loss + loss

def CLS_loss(label, logits):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))
    return loss

# Subspace Minimax Entropy loss
def SME_loss(features, Centroids, numSeg):

    x = tf.split(features, numSeg, 1)
    y = tf.split(Centroids, numSeg, 0)

    for i in range(numSeg):
        tmp = tf.expand_dims(tf.matmul(x[i], y[i]), axis=-1)
        if i==0:
            logits = tmp
        else:
            logits = tf.concat([logits, tmp], axis=2)

    logits = SoftMax(tf.reduce_mean(logits, axis=2), axis=1)
    loss = tf.reduce_mean(tf.reduce_sum(logits*(tf.log(logits + 1e-5)), 1))
    return loss


def avr_loss(diff, n_book,bn_word,batch_size):
    p = tf.split(diff, n_book, 1)
    loss=0.0
    for i in range(n_book):
        tem_loss=tf.reduce_sum(p[i],0)/batch_size
#         tem_loss=tf.square(tem_loss-1/10)#效果不好
        tem_loss=tf.square(tem_loss-1/pow(2,bn_word))#不一定要平方。三次方？ 
#         tem_loss=tf.square(tem_loss)  
        loss=loss+tf.reduce_sum(tem_loss)
    loss=loss/n_book
    return loss


def _pairwise_distances(embeddings_x,embeddings_q, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings_x, tf.transpose(embeddings_q))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    # Combine the two masks
    mask = tf.logical_and(indices_not_equal, labels_equal)

    return mask


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask = tf.logical_not(labels_equal)

    return mask


def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)


    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask


def batch_all_triplet_loss(labels, embeddings_x,embeddings_q, margin=0.5, squared=False):
    """
    Build the triplet loss over a batch of embeddings.
    We generate all the valid triplets and average the loss over the positive ones.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # embeddings_x = tf.nn.l2_normalize(embeddings_x, axis=1)
    # embeddings_q = tf.nn.l2_normalize(embeddings_q, axis=1)
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings_x,embeddings_q, squared=squared)

    # shape (batch_size, batch_size, 1)
    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels)
    mask = tf.to_float(mask)
    triplet_loss = tf.multiply(mask, triplet_loss)

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets


def batch_hard_triplet_loss(labels, embeddings_x,embeddings_q, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.
    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings_x,embeddings_q,  squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = tf.to_float(mask_anchor_positive)

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

    # shape (batch_size, 1)
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)
    tf.summary.scalar("hardest_positive_dist", tf.reduce_mean(hardest_positive_dist))

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = tf.to_float(mask_anchor_negative)

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)
    tf.summary.scalar("hardest_negative_dist", tf.reduce_mean(hardest_negative_dist))

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)

    # Get final mean triplet loss
    triplet_loss = tf.reduce_mean(triplet_loss)

    return triplet_loss