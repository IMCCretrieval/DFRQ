# from utils.GPQ_network import *
from unseen_utils.Functions import *
from unseen_utils import cifar_unseen as ci10
from unseen_utils.RetrievalTest import *

from datetime import datetime
import argparse


with tf.variable_scope("placeholder"):
    x = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='x')
    x_T = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='x_T')
    label = tf.placeholder(tf.float32, shape=[None, n_CLASSES], name='label')
    label_Mat = tf.placeholder(tf.float32, shape=[None, None], name='label_Mat')
    training_flag = tf.placeholder(tf.bool, name='training_flag')
    global_step = tf.placeholder(tf.float32, name='global_step')

def data_deprocessing(x_):

    #Return to original data
    x_ = np.squeeze(x_)

    x_[:, :, :, 0] = x_[:, :, :,0]* 63.01 + 125.642
    x_[:, :, :, 1] = x_[:, :, :,1]* 62.157 + 123.738
    x_[:, :, :, 2] = x_[:, :, :,2]* 66.94 + 114.46

    return x_
def run():

    print("num_Codewords: 2^%d, num_Codebooks: %d, Bits: %d, code_len: %d" % (bn_word, n_book, n_bits, len_code))
    Source_x, Source_y, Target_x, Gallery_x,Gallery_y, Query_x, Query_y = ci10.prepare_data(data_dir, True)
    np.save("set2_label.npy",Gallery_y)
    Target_x_255 = np.copy(Target_x)

    Target_x_255=data_deprocessing(Target_x_255)  #fft


    Net = GPQ(training=training_flag)

    feature_S = Net.F(x)
    feature_T = Net.F(x_T)
    # feature_T = flip_gradient(Net.F(x_T))

    feature_S = Intra_Norm(feature_S, n_book)
    feature_T = Intra_Norm(feature_T, n_book)

    descriptor_S,_ = Soft_Assignment(Net.Z, feature_S, n_book, alpha)
    descriptor_T,tmp_T = Soft_Assignment(Net.Z, feature_T, n_book, alpha)
    # logits_S = Net.C(feature_S * beta, tf.transpose(Prototypes) * beta)

    hash_loss = N_PQ_loss(labels_Similarity=label_Mat, embeddings_x=feature_S, embeddings_q=descriptor_S, n_book=n_book)


    '''
    默认features_q 为 descriptor_T
    '''
    if eta == 0.0:
        av_loss = avr_loss(tmp_T,n_book,bn_word,batch_size) 
        cost =  hash_loss+ +eta_av*av_loss
    else:
        ranking_loss = self_ranking_loss(features_x=feature_T,features_q=descriptor_T,batch_size=unlabel_batch_size,count=self_count)
        av_loss = avr_loss(tmp_T,n_book,bn_word,batch_size) 
        cost =  hash_loss+ eta*ranking_loss+eta_av*av_loss

    pretrained_mat = scipy.io.loadmat(ImagNet_pretrained_path)

    var_F = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Fixed_VGG')

    decayed_lr = tf.train.exponential_decay(0.0002, global_step, 100, 0.98, staircase=True)
    train_op = tf.train.AdamOptimizer(learning_rate=decayed_lr, beta1=0.5).minimize(loss=cost)

    saver = tf.train.Saver(tf.global_variables())

    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())

        print("Load ImageNet2012 pretrained model")
        for i in range(len(var_F) - 2):
            sess.run(var_F[i].assign(np.squeeze(pretrained_mat[var_F[i].name])))

        total_iter = 0
  
        for epoch in range(1, total_epochs + 1):

            if epoch == 1:

                label_Similarity=label_Sim_Generate(Gallery_y,Query_y)
                num_S = np.shape(Source_x)[0]
                num_T = np.shape(Target_x_255)[0]

                iteration = int(num_S / batch_size)

            if epoch % save_term == 0:
                print('Model saved at %d'%(epoch))
                model_load_path='/home/lipd/DFRQ/models/final_model/24_bits_set2_cifar.ckpt'
                saver.restore(sess, model_load_path)
                mAP = PQ_retrieval(sess, x, training_flag, Net.F(x), Net.Z, n_book, Gallery_x, Query_x, label_Similarity, True, TOP_K=num_T)
                print( " mAP: %.4f"%(mAP))

class GPQ():
    def __init__(self, training):
        self.training = training
        self.Z = tf.get_variable('Z', [intn_word, len_code * n_book], dtype=tf.float32,
                                   initializer=initializer, trainable=True)#12*12=144-d  16 144
    #Feature Extractor
    def F(self, input_x):
        with tf.variable_scope('Fixed_VGG', reuse=tf.AUTO_REUSE):
            x = conv_layer(input_x, filter=64, kernel=[3, 3], stride=1, layer_name='conv0')
            x = Batch_Normalization(x, training=self.training, scope='batch0')
            x = Relu(x)
            x = conv_layer(x, filter=64, kernel=[3, 3], stride=1, layer_name='conv0-1')
            x = Batch_Normalization(x, training=self.training, scope='batch0-1')
            x = Relu(x)
            x = Max_Pooling(x, pool_size=[2, 2], stride=2)

            x = conv_layer(x, filter=128, kernel=[3, 3], stride=1, layer_name='conv1')
            x = Batch_Normalization(x, training=self.training, scope='batch1')
            x = Relu(x)
            x = conv_layer(x, filter=128, kernel=[3, 3], stride=1, layer_name='conv1-1')
            x = Batch_Normalization(x, training=self.training, scope='batch1-1')
            x = Relu(x)
            x = Max_Pooling(x, pool_size=[2, 2], stride=2)

            x = conv_layer(x, filter=256, kernel=[3, 3], stride=1, layer_name='conv2')
            x = Batch_Normalization(x, training=self.training, scope='batch2')
            x = Relu(x)
            x = conv_layer(x, filter=256, kernel=[3, 3], stride=1, layer_name='conv2-1')
            x = Batch_Normalization(x, training=self.training, scope='batch2-1')
            x = Relu(x)
            x = conv_layer(x, filter=256, kernel=[3, 3], stride=1, layer_name='conv2-2')
            x = Batch_Normalization(x, training=self.training, scope='batch2-2')
            x = Relu(x)
            x = Max_Pooling(x, pool_size=[2, 2], stride=2)
            x_branch = Global_Average_Pooling(x)

            x = conv_layer(x, filter=512, kernel=[3, 3], stride=1, layer_name='conv3')
            x = Batch_Normalization(x, training=self.training, scope='batch3')
            x = Relu(x)
            x = conv_layer(x, filter=512, kernel=[3, 3], stride=1, layer_name='conv3-1')
            x = Batch_Normalization(x, training=self.training, scope='batch3-1')
            x = Relu(x)
            x = conv_layer(x, filter=512, kernel=[3, 3], stride=1, layer_name='conv3-2')
            x = Batch_Normalization(x, training=self.training, scope='batch3-2')
            x = Relu(x)
            
            x = Global_Average_Pooling(x)
            x = tf.concat([x, x_branch], 1)
            
            x = Linear(x, len_code * n_book, layer_name='feature_vector')

        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--book_num", type=int, default=6)
    parser.add_argument("--word_num", type=int, default=4)
    parser.add_argument("--len_code", type=int, default=12)
    # parser.add_argument("--img_size", type=int, default=32)
    # parser.add_argument("--batch", type=int, default=500)
    # parser.add_argument("--beta", type=int, default=1)
    parser.add_argument("--eta", default=0.5, type=float) #1.0
    parser.add_argument("--eta_av", default=0.8, type=float)
    parser.add_argument("--save_term", type=int, default=60)
    opts = parser.parse_args()

    SEED =  0  #0-》0.885
    random.seed(SEED)
    np.random.seed(SEED)
    tf.set_random_seed(SEED)

    print(SEED)
    # Number of codebooks
    n_book = opts.book_num

    # Number of codewords=(2^bn_word)
    bn_word = opts.word_num

    global intn_word 
    intn_word = pow(2, bn_word)

    n_bits = n_book * bn_word

    # length of codeword
    len_code =opts.len_code

    # batch_size = opts.batch

    # save model for every save_term-th epoch
    save_term = opts.save_term
    eta_av=opts.eta_av  
    eta=opts.eta

    run()