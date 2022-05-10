from opts2 import *
from scipy.signal import convolve2d
import os.path
import cv2
from keras.callbacks import Callback, TensorBoard
import numpy as np

NUM_EPOCHS_PER_DECAY = 50  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.5  # Learning rate decay factor.
grad_clip = tf.constant(0.005, tf.float32)
FLAGS = tf.app.flags.FLAGS
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
mv_k = 51
mv_ks = 21
INITIAL_LEARNING_RATE = 0.0001
ite_decay = 200000 #80000 # 迭代次数，开始降低学习率


''' Callbacks '''
class HistoryCheckpoint(Callback):
    '''Callback that records events
        into a `History` object.
        It then saves the history after each epoch into a file.
        To read the file into a python dict:
            history = {}
            with open(filename, "r") as f:
                history = eval(f.read())
        This may be unsafe since eval() will evaluate any string
        A safer alternative:
        import ast
        history = {}
        with open(filename, "r") as f:
            history = ast.literal_eval(f.read())
    '''

    def __init__(self, filename):
        super(Callback, self).__init__()
        self.filename = filename

    def on_train_begin(self, logs={}):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs={}):
        self.epoch.append(epoch)
        for k, v in logs.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(v)

        with open(self.filename, "w") as f:
            f.write(str(self.history))


'''
Below is a modification to the TensorBoard callback to perform 
batchwise writing to the tensorboard, instead of only at the end
of the batch.
'''
class TensorBoardBatch(TensorBoard):
    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 batch_size=32,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None):
        super(TensorBoardBatch, self).__init__(log_dir,
                                               histogram_freq=histogram_freq,
                                               batch_size=batch_size,
                                               write_graph=write_graph,
                                               write_grads=write_grads,
                                               write_images=write_images,
                                               embeddings_freq=embeddings_freq,
                                               embeddings_layer_names=embeddings_layer_names,
                                               embeddings_metadata=embeddings_metadata)

        # conditionally import tensorflow iff TensorBoardBatch is created
        self.tf = __import__('tensorflow')
        self.global_step = 1

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = self.tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, self.global_step)
        self.global_step += 1

        self.writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = self.tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, self.global_step)

        self.global_step += 1
        self.writer.flush()


def train(total_loss, global_step):
  decay_steps = ite_decay
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  loss_averages_op = _add_loss_summaries(total_loss)
  with tf.control_dependencies([loss_averages_op]):
      opt = tf.train.AdamOptimizer(lr)
      grads = opt.compute_gradients(total_loss)
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())
  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')
  return train_op, loss_averages_op, lr
def trainorg(total_loss, global_step):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  #NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = ite_decay * FLAGS.batch_size
  #num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = ite_decay#int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  #lr = tf.random_uniform((), maxval=0.01)
  '''step1=tf.constant(5000.0)
  max_lr=tf.constant(0.0001)
  grad_clip = tf.constant(0.005, tf.float32)
  lr=tf.divide(tf.multiply(max_lr,tf.subtract(step1,tf.cast(tf.mod(global_step,tf.cast(step1,tf.int32)),tf.float32))),step1)'''
  tf.summary.scalar('learning_rate', lr)
  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)
  '''var_list=tf.trainable_variables()
  w_list=[]
  b_list=[]
  for i in range(0,len(var_list)):
      if(var_list[i].name[-3]=='w'):
          w_list.append(var_list[i])
      else :
          b_list.append(var_list[i])'''
  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
      #opt = tf.train.GradientDescentOptimizer(lr)
      #opt = tf.train.MomentumOptimizer(lr,momentum=0.9)
      opt = tf.train.AdamOptimizer(lr)
      #opt=tf.train.RMSPropOptimizer(lr)
      grads = opt.compute_gradients(total_loss)
      '''clip_grads = []
      ind=0
      for grad,var in grads:
          if(ind%2 == 0):
              #clip_grads.append((grad,var))
              clip_grads.append((tf.clip_by_value(grad, -grad_clip/lr,grad_clip/lr), var))
          else :
              clip_grads.append((tf.multiply(tf.constant(0.1,tf.float32),tf.clip_by_value(grad, -grad_clip / lr, grad_clip / lr)), var))
              #clip_grads.append((grad*0.1, var))
          ind += 1'''
            #clip_grads.append((tf.multiply(tf.constant(0.1),tf.clip_by_value(grad, -grad_clip / lr, grad_clip / lr)), var))
      #clip_grads = [(tf.clip_by_value(grad, -grad_clip/lr,grad_clip/lr), var) for grad, var in grads]

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
  #apply_gradient_op = opt.apply_gradients(clip_grads, global_step=global_step)
  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  '''for grad, var in grads:
    if grad:
      tf.summary.histogram(var.op.name + '/gradients', grad)'''

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op,loss_averages_op,lr
def trainft(total_loss, global_step):
  """small learning rate """
  decay_steps = ite_decay#int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
  INITIAL_LEARNING_RATE1 = 0.00005
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE1,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  with tf.control_dependencies([loss_averages_op]):
      opt = tf.train.AdamOptimizer(lr)
      grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)
  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())
  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')
  return train_op,loss_averages_op,lr

def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name +' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))
  return loss_averages_op
# 有 Batch Norm
def trainBN(total_loss, global_step):
    # Batch Norm
    NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 80000 * FLAGS.batch_size
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)  # 几个batch后降低学习率

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    # lr = tf.random_uniform((), maxval=0.01)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)
    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)   # 开启batch norm
    with tf.control_dependencies(update_ops):
        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')

    return train_op, lr # ,loss_averages_op

def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")
    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2

    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))

    return np.mean(np.mean(ssim_map))
def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)

def psnr(im1,im2):
    diff =np.float64(im1[:]) - np.float64(im2[:])
    rmse = np.sqrt(np.mean(diff**2))
    psnr = 20*np.log10(255/rmse)
    return psnr,rmse

def get_filenames(paths):
    filenames = []
    for path in paths:
        for root, dirs, files in os.walk(path):
            for f in files:
                filenames.append(os.path.join(root, f))
    return filenames
def get_files(paths):
    filenames = []
    for path in paths:

        for root, dirs, files in os.walk(path):
            for f in dirs:
                filenames.append(os.path.join(root, f))
    return filenames

def rgb2y(rgb):
    h,w,d=rgb.shape
    rgb=np.float32(rgb)/255.0
    y=rgb*(np.reshape([65.481, 128.553, 24.966],[1,1,3])/255.0)
    y=y[:,:,0]+y[:,:,1]+y[:,:,2]
    y=np.reshape(y,[h,w])+16/255.0
    return np.uint8(y*255+0.5)

def img_to_uint8(img):
    img = np.clip(img, 0, 255)
    return np.round(img).astype(np.uint8)

rgb_to_ycbcr = np.array([[65.481, 128.553, 24.966],
                         [-37.797, -74.203, 112.0],
                         [112.0, -93.786, -18.214]])

ycbcr_to_rgb = np.linalg.inv(rgb_to_ycbcr)

def rgb2ycbcr(img):
    """ img value must be between 0 and 255"""
    img = np.float64(img)
    img = np.dot(img, rgb_to_ycbcr.T) / 255.0
    img = img + np.array([16, 128, 128])
    return img

def ycbcr2rgb(img):
    """ img value must be between 0 and 255"""
    img = np.float64(img)
    img = img - np.array([16, 128, 128])
    img = np.dot(img, ycbcr_to_rgb.T) * 255.0
    return img

def disploss(modelOutput, groundTruth):
        mask = groundTruth > 0
        g = tf.boolean_mask(groundTruth,mask)
        m = tf.boolean_mask(modelOutput,mask)
        count = tf.cast(tf.count_nonzero(mask), tf.float32)
        loss_all = tf.reduce_sum(tf.sqrt(tf.pow(g - m, 2) + 4) / 2 - 1) / count
        return loss_all
def MSEloss(out,label):
    sub=tf.subtract(out,label)
    l2 = tf.pow(sub, 2)
    MSE_loss = tf.reduce_sum(tf.reduce_sum(l2,axis=[1,2,3]),name='MSELoss')
    tf.add_to_collection('losses', MSE_loss)
    return tf.add_n(tf.get_collection('losses'),name="total_loss")
# (res, lr_align_ref, cl,ll)
def MSEOF(out1,out2,label1,l):
    sub1=tf.subtract(out1,label1)
    l2 = tf.pow(sub1, 2)
    MSE_loss1 = tf.reduce_sum(tf.reduce_sum(l2,axis=[1,2,3]),name='MSELossleft')
    sub2 = tf.subtract(out2,l)
    l22 = tf.pow(sub2, 2)
    MSE_loss2 = tf.reduce_sum(tf.reduce_sum(l22, axis=[1, 2, 3]), name='MSELossright_warp')
    MSE_loss=MSE_loss1 + 0.5*MSE_loss2
    tf.add_to_collection('losses', MSE_loss)
    return tf.add_n(tf.get_collection('losses'),name="total_loss")
# srl,srr, cl,cr
def MSELR(out1,out2,label1,label2):
    sub1=tf.subtract(out1,label1)
    l2 = tf.pow(sub1, 2)
    MSE_loss1 = tf.reduce_sum(tf.reduce_sum(l2,axis=[1,2,3]),name='MSELossleft')
    sub2 = tf.subtract(out2, label2)
    l22 = tf.pow(sub2, 2)
    MSE_loss2 = tf.reduce_sum(tf.reduce_sum(l22, axis=[1, 2, 3]), name='MSELossright')
    MSE_loss=MSE_loss1 + MSE_loss2
    tf.add_to_collection('losses', MSE_loss)
    return tf.add_n(tf.get_collection('losses'),name="total_loss")
# srl, srr, res, cl, cr
def MSEALL(out1,out2,outL,label1,label2):
    sub1=tf.subtract(out1,label1)
    l2 = tf.pow(sub1, 2)
    MSE_loss1 = tf.reduce_sum(tf.reduce_sum(l2,axis=[1,2,3]),name='MSELossleft')
    sub2 = tf.subtract(out2, label2)
    l22 = tf.pow(sub2, 2)
    MSE_loss2 = tf.reduce_sum(tf.reduce_sum(l22, axis=[1, 2, 3]), name='MSELossright')
    sub3 = tf.subtract(outL, label1)
    l23 = tf.pow(sub3, 2)
    MSE_loss3 = tf.reduce_sum(tf.reduce_sum(l23, axis=[1, 2, 3]), name='MSELossleftfine')
    MSE_loss=0.5*MSE_loss1 + 0.5*MSE_loss2 + MSE_loss3
    tf.add_to_collection('lossesl', MSE_loss1)
    tf.add_to_collection('lossesr', MSE_loss2)
    tf.add_to_collection('losseslf', MSE_loss3)
    tf.add_to_collection('lossall', MSE_loss)
    return tf.add_n(tf.get_collection('lossall'),name="total_loss")
# srl, srr, resl,resr, cl, cr
def MSEALLLR(out1,out2,outL,outR,label1,label2):
    sub1=tf.subtract(out1,label1)
    l2 = tf.pow(sub1, 2)
    MSE_loss1 = tf.reduce_sum(tf.reduce_sum(l2,axis=[1,2,3]),name='MSELossleft')
    sub2 = tf.subtract(out2, label2)
    l22 = tf.pow(sub2, 2)
    MSE_loss2 = tf.reduce_sum(tf.reduce_sum(l22, axis=[1, 2, 3]), name='MSELossright')
    sub3 = tf.subtract(outL, label1)
    l23 = tf.pow(sub3, 2)
    MSE_loss3 = tf.reduce_sum(tf.reduce_sum(l23, axis=[1, 2, 3]), name='MSELossleftfine')
    sub4 = tf.subtract(outR, label2)
    l24 = tf.pow(sub4, 2)
    MSE_loss4 = tf.reduce_sum(tf.reduce_sum(l24, axis=[1, 2, 3]), name='MSELossrightfine')
    MSE_loss=0.5*MSE_loss1 + 0.5*MSE_loss2 + MSE_loss3 + MSE_loss4
    tf.add_to_collection('lossesl', MSE_loss1)
    tf.add_to_collection('lossesr', MSE_loss2)
    tf.add_to_collection('losseslf', MSE_loss3)
    tf.add_to_collection('lossesrf', MSE_loss4)
    tf.add_to_collection('lossall', MSE_loss)
    return tf.add_n(tf.get_collection('lossall'),name="total_loss")
# srl, srr,lr_align_ref, res, cl, cr
def MSEALLOF(out1,out2,outrl,outL,label1,label2):
    sub1=tf.subtract(out1,label1)
    l2 = tf.pow(sub1, 2)
    MSE_loss1 = tf.reduce_sum(tf.reduce_sum(l2,axis=[1,2,3]),name='MSELossleft')
    sub2 = tf.subtract(out2, label2)
    l22 = tf.pow(sub2, 2)
    MSE_loss2 = tf.reduce_sum(tf.reduce_sum(l22, axis=[1, 2, 3]), name='MSELossright')

    subrl = tf.subtract(outrl, label1)
    l2rl = tf.pow(subrl, 2)
    MSE_loss1rl = tf.reduce_sum(tf.reduce_sum(l2rl, axis=[1, 2, 3]), name='MSELossleft')

    sub3 = tf.subtract(outL, label1)
    l23 = tf.pow(sub3, 2)
    MSE_loss3 = tf.reduce_sum(tf.reduce_sum(l23, axis=[1, 2, 3]), name='MSELossleftfine')

    MSE_loss=0.1*MSE_loss1 + 0.1*MSE_loss2 + MSE_loss3 + 0.1*MSE_loss1rl #0.05*MSE_loss1rl
    tf.add_to_collection('lossesl', MSE_loss1)
    tf.add_to_collection('lossesr', MSE_loss2)
    tf.add_to_collection('losseslf', MSE_loss3)
    tf.add_to_collection('lossall', MSE_loss)
    return tf.add_n(tf.get_collection('lossall'),name="total_loss")
def MSEALLOF2(out1,out2,outL,label1,label2):
    sub1=tf.subtract(out1,label1)
    l2 = tf.pow(sub1, 2)
    MSE_loss1 = tf.reduce_sum(tf.reduce_sum(l2,axis=[1,2,3]),name='MSELossleft')
    sub2 = tf.subtract(out2, label2)
    l22 = tf.pow(sub2, 2)
    MSE_loss2 = tf.reduce_sum(tf.reduce_sum(l22, axis=[1, 2, 3]), name='MSELossright')

    sub3 = tf.subtract(outL, label1)
    l23 = tf.pow(sub3, 2)
    MSE_loss3 = tf.reduce_sum(tf.reduce_sum(l23, axis=[1, 2, 3]), name='MSELossleftfine')

    MSE_loss= 0.5*MSE_loss1 + 0.5*MSE_loss2 + MSE_loss3 #0.05*MSE_loss1rl
    #MSE_loss= 0.1*MSE_loss1 + 0.1*MSE_loss2 + MSE_loss3 #0.05*MSE_loss1rl
    tf.add_to_collection('lossesl', MSE_loss1)
    tf.add_to_collection('lossesr', MSE_loss2)
    tf.add_to_collection('losseslf', MSE_loss3)
    tf.add_to_collection('lossall', MSE_loss)
    return tf.add_n(tf.get_collection('lossall'),name="total_loss")
def MSEALLOF2im(out1,out2,outL,outR,label1,label2): #sl, sr, resl,resr, cl, cr
    sub1=tf.subtract(out1,label1)
    l2 = tf.pow(sub1, 2)
    MSE_loss1 = tf.reduce_sum(tf.reduce_sum(l2,axis=[1,2,3]),name='MSELossleft')
    sub2 = tf.subtract(out2, label2)
    l22 = tf.pow(sub2, 2)
    MSE_loss2 = tf.reduce_sum(tf.reduce_sum(l22, axis=[1, 2, 3]), name='MSELossright')

    sub3 = tf.subtract(outL, label1)
    l23 = tf.pow(sub3, 2)
    MSE_loss3 = tf.reduce_sum(tf.reduce_sum(l23, axis=[1, 2, 3]), name='MSELossleftfine')
    sub4 = tf.subtract(outR, label2)
    l234 = tf.pow(sub4, 2)
    MSE_loss4 = tf.reduce_sum(tf.reduce_sum(l234, axis=[1, 2, 3]), name='MSELossrightfine')

    MSE_loss= 0.5*MSE_loss1 + 0.5*MSE_loss2 + MSE_loss3+ MSE_loss4 #
    tf.add_to_collection('lossall', MSE_loss)
    return tf.add_n(tf.get_collection('lossall'),name="total_loss")#,MSE_loss1,MSE_loss2,MSE_loss3,MSE_loss4
# sry, src, cly, clc
def MSECV(out1,out2,label1,label2):
    sub1=tf.subtract(out1,label1)
    l2 = tf.pow(sub1, 2)
    MSE_loss1 = tf.reduce_sum(tf.reduce_sum(l2,axis=[1,2,3]),name='MSELossleft')
    sub2 = tf.subtract(out2, label2)
    l22 = tf.pow(sub2, 2)
    MSE_loss2 = tf.reduce_sum(tf.reduce_sum(l22, axis=[1, 2, 3]), name='MSELossright')

    MSE_loss= MSE_loss1 + MSE_loss2
    tf.add_to_collection('lossesl', MSE_loss1)
    tf.add_to_collection('lossesr', MSE_loss2)
    tf.add_to_collection('lossall', MSE_loss)
    return tf.add_n(tf.get_collection('lossall'),name="total_loss")

def gaussian_kernel_2d_opencv(sigma=0.0):
    kx = cv2.getGaussianKernel(15, sigma)
    ky = cv2.getGaussianKernel(15, sigma)
    return np.multiply(kx, np.transpose(ky))