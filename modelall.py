from opts2 import *
import keras
from keras.layers import Input, Conv2DTranspose, Concatenate, Conv2D, LeakyReLU, ELU, MaxPooling2D, Flatten, Dense, Dropout, normalization
import tensorlayer as tl
from tensorlayer.layers import *

fm_n=64
init_value2=[]
df_dim=64

##### ===================== tensorflow =========================== ##
def SR_localbic(l,b,fsize=128,scale = 4,small=True,Y=0):# 20 layer
    with tf.variable_scope("pret_ed",reuse =tf.AUTO_REUSE) as scope:
        x = conv2d(l, fsize, 3, 3, name='in')
        # Store the output of the first convolution to add later
        conv_1 = x
        for i in range(6):
            x = ResBlock_edsr(x, name='res' + str(i), f_size=fsize)
        x = conv2d(x, fsize, 3, 3, name='conv1')
        x += conv_1

        xim = lrelu(conv2d(x, fsize, 3, 3, name='convim'))

        xf = lrelu(conv2d(x, fsize, 3, 3, name='convf1'))
        xf1 = lrelu(conv2d(xf, fsize, 3, 3, name='convf2'))

        f = concat([conv_1,xim,xf1],3)
        f = conv2d(f, fsize, 3, 3, name='convout0')
        f = f+conv_1
        # Upsample output of the convolution
        x = deconv_anysize(f, 'deconv1', fsize, ksize=8, stride=scale)
        if Y==0:
            output = conv2d(x, 3, 3, 3, name='convout')
        else:
            output = conv2d(x, 1, 3, 3, name='convout')
        if b != None:
            output = output + b

        if small=='both':
            return xf1, x, output
        if small:
            return xf1, output  # small linVSR feature
        else:
            return x, output # big feature


def SR_ctxLeft(sl, sr, fl, f_size=128,small=False,s = 4): # 22
    with tf.variable_scope("context", reuse=tf.AUTO_REUSE) as scope:
        # left right image 交叉传递feature
        if small == True:# 利用smallfeature
            xl = lrelu(conv2d(tf.space_to_depth(sl, s, 'left', "NHWC") , f_size, 3, 3, name='in'))
            xr = lrelu(conv2d(tf.space_to_depth(sr, s, 'right', "NHWC") , f_size, 3, 3, name='inr'))
        else:
            xl = lrelu(conv2d(sl, f_size, 3, 3, name='in'))
            xr = lrelu(conv2d(sr, f_size, 3, 3, name='inr'))
        cat1 = concat([xl,xr],3)

        xl = lrelu(conv2d(cat1, f_size, 3, 3, name='l1'))
        xl = ResBlock_edsr(xl, name='res1', f_size=f_size)

        xr = lrelu(conv2d(cat1, f_size, 3, 3, name='r1'))
        xr = ResBlock_edsr(xr, name='resr1', f_size=f_size)
        cat2 = concat([xl, xr],3)

        xl = lrelu(conv2d(cat2, f_size, 3, 3, name='l2'))
        xl = ResBlock_edsr(xl, name='res2', f_size=f_size)

        xr = lrelu(conv2d(cat2, f_size, 3, 3, name='r2'))
        xr = ResBlock_edsr(xr, name='resr2', f_size=f_size)
        x = lrelu(conv2d(concat([xl, xr], 3), f_size, 3, 3, name='catconv'))

        # 级联image+feature
        if small == True:# 利用smallfeature
            im = conv2d(tf.space_to_depth(sl, s, 'left', "NHWC"), f_size, 3, 3, name='imin')
        else:
            im = conv2d(sl, f_size, 3, 3, name='imin')# im = conv2d(concat([sl,sr],3), f_size, 3, 3, name='imin')
        im = ResBlock_edsr(im, name='imres1', f_size=f_size)
        f = conv2d(concat([im,x,fl],3), f_size, 3, 3, name='fin')# f = conv2d(concat([x,fl,fr],3), f_size, 3, 3, name='fin')
        f = ResBlock_edsr(f, name='fres1', f_size=f_size)
        x = im + f
        if small == True:  # 利用smallfeature
            x = deconv_anysize(x, 'deconv1', f_size, ksize=8, stride=s)
            x = conv2d(x, 3, 3, 3, name='convout')
        else:
            x = conv2d(x, 3, 3, 3, name='convout')
        output = x + sl
        return output
def SR_ctxLeftRight(sl, sr, fl, fr, f_size=128, s = 4,small=False, feature=False): # 19
    with tf.variable_scope("context", reuse=tf.AUTO_REUSE) as scope:
        if small: #  spacetodepth  time??  PSNR??s`
            sl1 = tf.space_to_depth(sl, s, 'left', "NHWC")  # scale=4 blockSize=2 # 2的倍数
            sr1 = tf.space_to_depth(sr, s, 'right', "NHWC")
        else:  # left right image 交叉传递feature
            sl1 = sl
            sr1 = sr
        xl = lrelu(conv2d(sl1, f_size, 3, 3, name='in'))
        xr = lrelu(conv2d(sr1, f_size, 3, 3, name='inr'))
        cat1 = concat([xl, xr],3)

        xl = lrelu(conv2d(cat1, f_size, 3, 3, name='l1'))
        xl = ResBlock_edsr(xl, name='res1', f_size=f_size)

        xr = lrelu(conv2d(cat1, f_size, 3, 3, name='r1'))
        xr = ResBlock_edsr(xr, name='resr1', f_size=f_size)
        cat2 = concat([xl,xr],3)

        xl = lrelu(conv2d(cat2, f_size, 3, 3, name='l2'))
        xl = ResBlock_edsr(xl, name='res2', f_size=f_size)

        xr = lrelu(conv2d(cat2, f_size, 3, 3, name='r2'))
        xr = ResBlock_edsr(xr, name='resr2', f_size=f_size)
        cat3 = concat([xl, xr], 3)
        x = lrelu(conv2d(cat3, f_size, 3, 3, name='catconv'))

        if small:# 级联image+feature
            im = conv2d(sl1, f_size, 3, 3, name='imin')
            imr = conv2d(sr1, f_size, 3, 3, name='iminr')
            im = ResBlock_edsr(im, name='imres1', f_size=f_size)
            imr = ResBlock_edsr(imr, name='imresr', f_size=f_size)
            fl = conv2d(concat([im, x, fl], 3), f_size, 3, 3, name='fin')
            fr = conv2d(concat([imr, x, fr], 3), f_size, 3, 3, name='finr')
            fl = ResBlock_edsr(fl, name='fres1', f_size=f_size)
            fr = ResBlock_edsr(fr, name='fresr', f_size=f_size)
            xl = im + fl
            xl = deconv_anysize(xl, 'deconv1', f_size, ksize=8, stride=s)
            xl = conv2d(xl, 3, 3, 3, name='convout')
            output = xl + sl
            xr = imr + fr
            xr = deconv_anysize(xr, 'deconv1r', f_size, ksize=8, stride=s)
            xr = conv2d(xr, 3, 3, 3, name='convoutr')
            outputr = xr + sr
        else:
            # im = conv2d(concat([sl,sr],3), f_size, 3, 3, name='imin')
            im = conv2d(sl1, f_size, 3, 3, name='imin')
            im = ResBlock_edsr(im, name='imres1', f_size=f_size)
            # f = conv2d(concat([x,fl,fr],3), f_size, 3, 3, name='fin')
            f = conv2d(concat([im, x, fl], 3), f_size, 3, 3, name='fin')
            f = ResBlock_edsr(f, name='fres1', f_size=f_size)
            fl = f
            x = im + f
            x = conv2d(x, 3, 3, 3, name='convout')
            output = x + sl

            imr = conv2d(sr1, f_size, 3, 3, name='iminr')
            imr = ResBlock_edsr(imr, name='imresr', f_size=f_size)
            f = conv2d(concat([imr, x, fr], 3), f_size, 3, 3, name='finr') # x被left处理了！！！
            f = ResBlock_edsr(f, name='fresr', f_size=f_size)
            fr = f
            x = imr + f
            x = conv2d(x, 3, 3, 3, name='convoutr')
            outputr = x + sr
        if feature:
            return output,outputr, fl, fr
        else:
            return output,outputr

##### ===================== tensorflow =========================== ##
def SR_localbicps(l,b,fsize=128,small=True, ps=2):# 20 layer
    with tf.variable_scope("pret_ed",reuse =tf.AUTO_REUSE) as scope:
        x = conv2d(l, fsize, 3, 3, name='in')
        # Store the output of the first convolution to add later
        conv_1 = x
        for i in range(6):
            x = ResBlock_edsr(x, name='res' + str(i), f_size=fsize)
        x = conv2d(x, fsize, 3, 3, name='conv1')
        x += conv_1

        xim = lrelu(conv2d(x, fsize, 3, 3, name='convim'))

        xf = lrelu(conv2d(x, fsize, 3, 3, name='convf1'))
        xf1 = lrelu(conv2d(xf, fsize, 3, 3, name='convf2'))

        f = concat([conv_1,xim,xf1],3)
        f = conv2d(f, fsize, 3, 3, name='convout0')
        f = f+conv_1
        # Upsample output of the convolution
        if ps ==1:
            f = conv2d(f, 12, 3, 3, name='deconv0')
            x = pixel_shift(f, 2, 3)
            x = conv2d(x, 12, 3, 3, name='deconv1')
            x = pixel_shift(x, 2, 3)
        else:
            f = conv2d(f, 3, 3, 3, name='deconv0')
            x = tf.image.resize_bicubic(f,[f.get_shape()[1].value * 4, f.get_shape()[2].value * 4])
            # bic_image = tf.image.resize_images(lr_image, [inh1, inw1], method=2)

        output = conv2d(x, 3, 3, 3, name='deconv2')
        # x = deconv_anysize(f, 'deconv1', fsize, ksize=8, stride=scale)
        if b != None:
            output = output + b
        if small:
            return xf1, output  # small linVSR feature
        else:
            return x, output # big feature

def SR_ctxLeftps(sl, sr, fl, f_size=128,small=False,s = 4, ps=2): # 22
    with tf.variable_scope("context", reuse=tf.AUTO_REUSE) as scope:
        # left right image 交叉传递feature
        if small == True:# 利用smallfeature
            xl = lrelu(conv2d(tf.space_to_depth(sl, s, 'left', "NHWC") , f_size, 3, 3, name='in'))
            xr = lrelu(conv2d(tf.space_to_depth(sr, s, 'right', "NHWC") , f_size, 3, 3, name='inr'))
        else:
            xl = lrelu(conv2d(sl, f_size, 3, 3, name='in'))
            xr = lrelu(conv2d(sr, f_size, 3, 3, name='inr'))
        cat1 = concat([xl,xr],3)

        xl = lrelu(conv2d(cat1, f_size, 3, 3, name='l1'))
        xl = ResBlock_edsr(xl, name='res1', f_size=f_size)

        xr = lrelu(conv2d(cat1, f_size, 3, 3, name='r1'))
        xr = ResBlock_edsr(xr, name='resr1', f_size=f_size)
        cat2 = concat([xl, xr],3)

        xl = lrelu(conv2d(cat2, f_size, 3, 3, name='l2'))
        xl = ResBlock_edsr(xl, name='res2', f_size=f_size)

        xr = lrelu(conv2d(cat2, f_size, 3, 3, name='r2'))
        xr = ResBlock_edsr(xr, name='resr2', f_size=f_size)
        x = lrelu(conv2d(concat([xl, xr], 3), f_size, 3, 3, name='catconv'))

        # 级联image+feature
        if small == True:# 利用smallfeature
            im = conv2d(tf.space_to_depth(sl, s, 'left', "NHWC"), f_size, 3, 3, name='imin')
        else:
            im = conv2d(sl, f_size, 3, 3, name='imin')# im = conv2d(concat([sl,sr],3), f_size, 3, 3, name='imin')
        im = ResBlock_edsr(im, name='imres1', f_size=f_size)
        f = conv2d(concat([im,x,fl],3), f_size, 3, 3, name='fin')# f = conv2d(concat([x,fl,fr],3), f_size, 3, 3, name='fin')
        f = ResBlock_edsr(f, name='fres1', f_size=f_size)
        x = im + f
        if small == True:  # 利用smallfeature
            if ps==1:
                x = pixel_shift(x, 2, 1)
                x = conv2d(x, f_size, 3, 3, name='deconv1')
                x = pixel_shift(x, 2, 1)
                x = conv2d(x, f_size, 3, 3, name='deconv2')
                #  x = deconv_anysize(x, 'deconv1', f_size, ksize=8, stride=s)
            else:
                f = conv2d(f, 3, 3, 3, name='deconv0')
                x = tf.image.resize_bicubic(f, [f.get_shape()[1].value * 4, f.get_shape()[2].value * 4])

            x = conv2d(x, 3, 3, 3, name='convout')
        else:
            x = conv2d(x, 3, 3, 3, name='convout')
        output = x + sl
        return output
def SR_ctxLeftRightps(sl, sr, fl, fr, f_size=128, s = 4,small=False, ps=2): # 19
    with tf.variable_scope("context", reuse=tf.AUTO_REUSE) as scope:
        if small: #  spacetodepth  time??  PSNR??s`
            sl1 = tf.space_to_depth(sl, s, 'left', "NHWC")  # scale=4 blockSize=2 # 2的倍数
            sr1 = tf.space_to_depth(sr, s, 'right', "NHWC")
        else:  # left right image 交叉传递feature
            sl1 = sl
            sr1 = sr
        xl = lrelu(conv2d(sl1, f_size, 3, 3, name='in'))
        xr = lrelu(conv2d(sr1, f_size, 3, 3, name='inr'))
        cat1 = concat([xl, xr],3)

        xl = lrelu(conv2d(cat1, f_size, 3, 3, name='l1'))
        xl = ResBlock_edsr(xl, name='res1', f_size=f_size)

        xr = lrelu(conv2d(cat1, f_size, 3, 3, name='r1'))
        xr = ResBlock_edsr(xr, name='resr1', f_size=f_size)
        cat2 = concat([xl,xr],3)

        xl = lrelu(conv2d(cat2, f_size, 3, 3, name='l2'))
        xl = ResBlock_edsr(xl, name='res2', f_size=f_size)

        xr = lrelu(conv2d(cat2, f_size, 3, 3, name='r2'))
        xr = ResBlock_edsr(xr, name='resr2', f_size=f_size)
        cat3 = concat([xl, xr], 3)
        x = lrelu(conv2d(cat3, f_size, 3, 3, name='catconv'))

        if small:# 级联image+feature
            im = conv2d(sl1, f_size, 3, 3, name='imin')
            imr = conv2d(sr1, f_size, 3, 3, name='iminr')
            im = ResBlock_edsr(im, name='imres1', f_size=f_size)
            imr = ResBlock_edsr(imr, name='imresr', f_size=f_size)
            fl = conv2d(concat([im, x, fl], 3), f_size, 3, 3, name='fin')
            fr = conv2d(concat([imr, x, fr], 3), f_size, 3, 3, name='finr')
            fl = ResBlock_edsr(fl, name='fres1', f_size=f_size)
            fr = ResBlock_edsr(fr, name='fresr', f_size=f_size)
            xl = im + fl
            xr = imr + fr

            if ps ==1:
                f = conv2d(xl, 12, 3, 3, name='deconv0')
                x = pixel_shift(f, 2, 3)
                x = conv2d(x, 12, 3, 3, name='deconv1')
                x = pixel_shift(x, 2, 3)
                xl = conv2d(x, 3, 3, 3, name='deconv2')
                #  xl = deconv_anysize(xl, 'deconv1', f_size, ksize=8, stride=s)
                f = conv2d(xr, 12, 3, 3, name='deconv0')
                x = pixel_shift(f, 2, 3)
                x = conv2d(x, 12, 3, 3, name='deconv1r')
                x = pixel_shift(x, 2, 3)
                xr = conv2d(x, 3, 3, 3, name='deconv1r2')
            else:
                f = conv2d(xl, 3, 3, 3, name='deconv0')
                xl = tf.image.resize_bicubic(f, [f.get_shape()[1].value * 4, f.get_shape()[2].value * 4])
                fr = conv2d(xr, 3, 3, 3, name='deconv0')
                xr = tf.image.resize_bicubic(fr, [fr.get_shape()[1].value * 4, f.get_shape()[2].value * 4])

            output = xl + sl
            outputr = xr + sr
        else:
            im = conv2d(sl1, f_size, 3, 3, name='imin')
            im = ResBlock_edsr(im, name='imres1', f_size=f_size)
            f = conv2d(concat([im, x, fl], 3), f_size, 3, 3, name='fin')
            f = ResBlock_edsr(f, name='fres1', f_size=f_size)
            x = im + f
            x = conv2d(x, 3, 3, 3, name='convout')
            output = x + sl
            imr = conv2d(sr1, f_size, 3, 3, name='iminr')
            imr = ResBlock_edsr(imr, name='imresr', f_size=f_size)
            f = conv2d(concat([imr, x, fr], 3), f_size, 3, 3, name='finr')# x被left处理了！！！
            f = ResBlock_edsr(f, name='fresr', f_size=f_size)
            x = imr + f
            x = conv2d(x, 3, 3, 3, name='convoutr')
            outputr = x + sr
        return output,outputr

def ResBlock_edsr(h0,name,f_size=fm_n, kernel_size=3):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        h1=relu(conv2d(h0,f_size,kernel_size,kernel_size,name='conv1',init_value=init_value2))
        h2=conv2d(h1,f_size,kernel_size,kernel_size,name='conv2',init_value=init_value2)
        h4=tf.add(h2,h0)
        return h4
##########################----------Stereo-----------###################
def modelIQA(left_image, right_image, name='SRIQA', anyshape=1):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        # left image = Input(shape=(32, 32, 3))
        # conv1
        left_conv1 = conv2d(left_image, 32, 3, 3, name='conv1_left')
        left_elu1 = tf.nn.elu(left_conv1)
        left_pool1 = tf.nn.max_pool(left_elu1, (1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
        # conv2
        left_conv2 = conv2d(left_pool1, 32, 3, 3, name='conv2_left')
        left_elu2 = tf.nn.elu(left_conv2)
        left_pool2 = tf.nn.max_pool(left_elu2, (1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
        # conv3
        left_conv3 = conv2d(left_pool2, 64, 3, 3, name='conv3_left')
        left_elu3 = tf.nn.elu(left_conv3)
        # conv4
        left_conv4 = conv2d(left_elu3, 64, 3, 3, name='conv4_left')
        left_elu4 = tf.nn.elu(left_conv4)
        # conv5
        left_conv5 = conv2d(left_elu4, 128, 3, 3, name='conv5_left')
        left_elu5 = tf.nn.elu(left_conv5)
        if anyshape == 1:
            shape_previous = left_elu5.get_shape()
            left_pool5 = tf.nn.max_pool(left_elu5, (1, shape_previous[0], shape_previous[1], 1),
                            strides=(1, shape_previous[0], shape_previous[1], 1), padding='SAME')
        else:
            left_pool5 = tf.nn.max_pool(left_elu5, (1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
        # fc6
        left_flat6 = tf.layers.flatten(left_pool5)
        left_fc6 = tf.layers.dense(left_flat6, 512)
        left_elu6 = tf.nn.elu(left_fc6)
        left_drop6 = tf.layers.dropout(left_elu6, 0.35)
        # fc7
        left_fc7 = tf.layers.dense(left_drop6, 512)
        left_elu7 = tf.nn.elu(left_fc7)
        left_drop7 = tf.layers.dropout(left_elu7, 0.5)

        # right image # right_image = Input(shape=(32, 32, 3))
        # conv1
        right_conv1 = conv2d(right_image, 32, 3, 3, name='conv1_right')
        right_elu1 = tf.nn.elu(right_conv1)
        right_pool1 = tf.nn.max_pool(right_elu1, (1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
        # conv2
        right_conv2 = conv2d(right_pool1, 32, 3, 3, name='conv2_right')
        right_elu2 = tf.nn.elu(right_conv2)
        right_pool2 = tf.nn.max_pool(right_elu2, (1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
        # conv3
        right_conv3 = conv2d(right_pool2, 64, 3, 3, name='conv3_right')
        right_elu3 = tf.nn.elu(right_conv3)
        # conv4
        right_conv4 = conv2d(right_elu3, 64, 3, 3, name='conv4_right')
        right_elu4 = tf.nn.elu(right_conv4)
        # conv5
        right_conv5 = conv2d(right_elu4, 128, 3, 3, name='conv5_right')
        right_elu5 = tf.nn.elu(right_conv5)
        if anyshape == 1:
            shape_previous = right_elu5.get_shape()
            right_pool5 = tf.nn.max_pool(right_elu5, (1, shape_previous[0], shape_previous[1], 1),
                                        strides=(1, shape_previous[0], shape_previous[1], 1), padding='SAME')
        else:
            right_pool5 = tf.nn.max_pool(right_elu5, (1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
        # fc6
        right_flat6 = tf.layers.flatten(right_pool5)
        right_fc6 = tf.layers.dense(right_flat6, 512)
        right_elu6 = tf.nn.elu(right_fc6)
        right_drop6 = tf.layers.dropout(right_elu6, 0.35)
        # fc7
        right_fc7 = tf.layers.dense(right_drop6, 512)
        right_elu7 = tf.nn.elu(right_fc7)
        right_drop7 = tf.layers.dropout(right_elu7, 0.5)

        # concatenate1
        add_conv2 = left_conv2 + right_conv2
        subtract_conv2 = left_conv2 - right_conv2
        fusion1_conv2 = concat([add_conv2, subtract_conv2], axis=-1)
        fusion1_elu2 = tf.nn.elu(fusion1_conv2)
        fusion1_pool2 = tf.nn.max_pool(fusion1_elu2, (1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
        # conv3
        fusion1_conv3 = conv2d(fusion1_pool2, 64, 3, 3, name= 'conv3_fusion1')
        fusion1_elu3 = tf.nn.elu(fusion1_conv3)
        # conv4
        fusion1_conv4 = conv2d(fusion1_elu3, 64, 3, 3, name='conv4_fusion1')
        fusion1_elu4 = tf.nn.elu(fusion1_conv4)
        # conv5
        fusion1_conv5 = conv2d(fusion1_elu4, 128, 3, 3, name='conv5_fusion1')
        fusion1_elu5 = tf.nn.elu(fusion1_conv5)
        fusion1_pool5 = tf.nn.max_pool(fusion1_elu5, (1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
        # fc6
        fusion1_flat6 = tf.layers.flatten(fusion1_pool5)
        fusion1_fc6 = tf.layers.dense(fusion1_flat6, 512)
        fusion1_elu6 = tf.nn.elu(fusion1_fc6)
        fusion1_drop6 = tf.layers.dropout(fusion1_elu6, 0.35)
        # fc7
        fusion1_fc7 = tf.layers.dense(fusion1_drop6, 512)
        fusion1_elu7 = tf.nn.elu(fusion1_fc7)
        fusion1_drop7 = tf.layers.dropout(fusion1_elu7, 0.5)

        # concatenate2
        add_conv5 = left_conv5 + right_conv5
        subtract_conv5 = left_conv5 - right_conv5
        fusion2_conv5 = concat([add_conv5, subtract_conv5], axis=-1)
        fusion2_elu5 = tf.nn.elu(fusion2_conv5)
        fusion2_pool5 = tf.nn.max_pool(fusion2_elu5, (1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
        # fc6
        fusion2_flat6 = tf.layers.flatten(fusion2_pool5)
        fusion2_fc6 = tf.layers.dense(fusion2_flat6, 512)
        fusion2_elu6 = tf.nn.elu(fusion2_fc6)
        fusion2_drop6 = tf.layers.dropout(fusion2_elu6, 0.35)
        # fc7
        fusion2_fc7 = tf.layers.dense(fusion2_drop6, 512)
        fusion2_elu7 = tf.nn.elu(fusion2_fc7)
        fusion2_drop7 = tf.layers.dropout(fusion2_elu7, 0.5)

        # concatenate3
        fusion3_drop7 = concat([left_drop7, right_drop7, fusion1_drop7, fusion2_drop7], axis=-1)
        # fc8
        fusion3_fc8 = tf.layers.dense(fusion3_drop7, 1024)
        # fc9
        predictions = tf.layers.dense(fusion3_fc8, 1)

        return predictions
def modelIQATL(left_image, right_image, leftf=None,rightf=None, df_dim=32, name='SRIQA', fe=0, reuse = tf.AUTO_REUSE):
    with tf.variable_scope(name, reuse=reuse) as scope:
        tl.layers.set_name_reuse(True)
        left_image = tl.layers.InputLayer(left_image, name='input/images')
        # conv1
        left_conv1 = tl.layers.Conv2d(left_image, df_dim, (3, 3), (1, 1), act=tf.nn.elu,
                        padding='SAME', W_init=w_init, name='conv1_left')
        left_pool1 = tl.layers.MaxPool2d(left_conv1, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='maxpool1')
        # conv2
        left_conv2 = tl.layers.Conv2d(left_pool1, df_dim, (3, 3), (1, 1), act=tf.nn.elu,
                        padding='SAME', W_init=w_init, name='conv2_left')
        left_pool2 = tl.layers.MaxPool2d(left_conv2, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='maxpool2')

        # conv3
        left_conv3 = tl.layers.Conv2d(left_pool2, df_dim*2, (3, 3), (1, 1), act=tf.nn.elu,
                                      padding='SAME', W_init=w_init, name='conv3_left')
        # conv4
        left_conv4 = tl.layers.Conv2d(left_conv3, df_dim*2, (3, 3), (1, 1), act=tf.nn.elu,
                                      padding='SAME', W_init=w_init, name='conv4_left')
        # conv5
        left_conv5 = tl.layers.Conv2d(left_conv4, df_dim*4, (3, 3), (1, 1), act=tf.nn.elu,
                        padding='SAME', W_init=w_init, name='conv5_left')
        if leftf is not None:
            left_conv5.outputs = leftf
        shape_previous = left_conv5.outputs.get_shape()
        left_pool5 = tl.layers.MaxPool2d(left_conv5, filter_size=(shape_previous[1], shape_previous[2]),
                            strides=(shape_previous[1], shape_previous[2]), padding='SAME', name='maxpool5')

        # fc6
        left_flat6 = tl.layers.FlattenLayer(left_pool5, name='flatten1')
        left_fc6 = tl.layers.DenseLayer(left_flat6, n_units=512, act=tf.nn.elu, name='fc1')
        left_drop6 = tl.layers.DropoutLayer(left_fc6, 0.35, is_fix=True, name='drop0')
        # fc7
        left_fc7 = tl.layers.DenseLayer(left_drop6, n_units=512, act=tf.nn.elu, name='fc2')
        left_drop7 = tl.layers.DropoutLayer(left_fc7, 0.5, is_fix=True, name='drop1')

        # conv1
        right_image = tl.layers.InputLayer(right_image, name='input/imagesright')
        # conv1
        right_conv1 = tl.layers.Conv2d(right_image, df_dim, (3, 3), (1, 1), act=tf.nn.elu,
                                      padding='SAME', W_init=w_init, name='conv1_right')
        right_pool1 = tl.layers.MaxPool2d(right_conv1, filter_size=(2, 2), strides=(2, 2), padding='SAME',
                                         name='maxpoolr1')
        # conv2
        right_conv2 = tl.layers.Conv2d(right_pool1, df_dim, (3, 3), (1, 1), act=tf.nn.elu, padding='SAME', W_init=w_init, name='conv2_right')
        right_pool2 = tl.layers.MaxPool2d(right_conv2, filter_size=(2, 2), strides=(2, 2), padding='SAME',
                                         name='maxpoolr2')

        # conv3
        right_conv3 = tl.layers.Conv2d(right_pool2, df_dim * 2, (3, 3), (1, 1), act=tf.nn.elu,
                                      padding='SAME', W_init=w_init, name='conv3_right')
        # conv4
        right_conv4 = tl.layers.Conv2d(right_conv3, df_dim * 2, (3, 3), (1, 1), act=tf.nn.elu,
                                      padding='SAME', W_init=w_init, name='conv4_right')
        # conv5
        right_conv5 = tl.layers.Conv2d(right_conv4, df_dim * 4, (3, 3), (1, 1), act=tf.nn.elu,
                                      padding='SAME', W_init=w_init, name='conv5_right')
        if rightf is not None:
            right_conv5.outputs = rightf
        shape_previous = right_conv5.outputs.get_shape()
        right_pool5 = tl.layers.MaxPool2d(right_conv5, filter_size=(shape_previous[1], shape_previous[2]), strides=(shape_previous[1], shape_previous[2]), padding='SAME',
                                         name='maxpoolr5')

        # fc6
        right_flat6 = tl.layers.FlattenLayer(right_pool5, name='flattenr1')
        right_fc6 = tl.layers.DenseLayer(right_flat6, n_units=512, act=tf.nn.elu, name='fcr1')
        right_drop6 = tl.layers.DropoutLayer(right_fc6, 0.35, is_fix=True, name='dropr2')
        # fc7
        right_fc7 = tl.layers.DenseLayer(right_drop6, n_units=512, act=tf.nn.elu, name='fcr2')
        right_drop7 = tl.layers.DropoutLayer(right_fc7, 0.5, is_fix=True, name='dropr3')

        #################### concatenate1 ################3
        add_conv2 = tl.layers.ElementwiseLayer([left_conv2, right_conv2], tf.add, name='sum1')
        subtract_conv2 = tl.layers.ElementwiseLayer([left_conv2, right_conv2], tf.subtract, name='sub1')
        fusion1_conv2 = tl.layers.ConcatLayer(layer=[add_conv2, subtract_conv2], concat_dim=-1, name='concat_layer')
        fusion1_elu2 = tl.layers.PReluLayer(fusion1_conv2, a_init=tf.constant_initializer(0.2), name='Prelu0')#.elu(fusion1_conv2)
        fusion1_pool2 = tl.layers.MaxPool2d(fusion1_elu2, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='maxpoolc1')
        # conv3
        fusion1_conv3 = tl.layers.Conv2d(fusion1_pool2, df_dim * 2, (3, 3), (1, 1), act=tf.nn.elu, padding='SAME', W_init=w_init, name='conv3_fusion1')

        # conv4
        fusion1_conv4 = tl.layers.Conv2d(fusion1_conv3, df_dim * 2, (3, 3), (1, 1), act=tf.nn.elu,
                                      padding='SAME', W_init=w_init, name='conv4_fusion1')
        # conv5
        fusion1_conv5 = tl.layers.Conv2d(fusion1_conv4, df_dim * 4, (3, 3), (1, 1), act=tf.nn.elu, padding='SAME', W_init=w_init, name='conv5_fusion1')
        shape_previous = fusion1_conv5.outputs.get_shape()
        fusion1_pool5 = tl.layers.MaxPool2d(fusion1_conv5, filter_size=(shape_previous[1], shape_previous[2]), strides=(shape_previous[1], shape_previous[2]), padding='SAME',
                                         name='maxpoolc2')
        # fc6
        fusion1_flat6 = tl.layers.FlattenLayer(fusion1_pool5, name='flattenc1')
        fusion1_fc6 = tl.layers.DenseLayer(fusion1_flat6, n_units=512, act=tf.nn.elu)
        fusion1_drop6 = tl.layers.DropoutLayer(fusion1_fc6, 0.35, is_fix=True, name='dropc4')
        # fc7
        fusion1_fc7 = tl.layers.DenseLayer(fusion1_drop6, n_units=512, act=tf.nn.elu, name='fcc1')
        fusion1_drop7 = tl.layers.DropoutLayer(fusion1_fc7, 0.5, is_fix=True, name='dropc5')

        # concatenate2
        add_conv5 = tl.layers.ElementwiseLayer([left_conv5, right_conv5], tf.add, name='sum2')
        subtract_conv5 = tl.layers.ElementwiseLayer([left_conv5, right_conv5], tf.subtract, name='sub2')
        fusion2_conv5 = tl.layers.ConcatLayer(layer=[add_conv5, subtract_conv5], concat_dim=-1, name='concat_layer2')
        fusion2_elu5 = tl.layers.PReluLayer(fusion2_conv5, a_init=tf.constant_initializer(0.2), name='Prelu1')  # tf.nn.elu(fusion1_conv5)
        shape_previous = fusion2_elu5.outputs.get_shape()
        fusion2_pool5 = tl.layers.MaxPool2d(fusion2_elu5, filter_size=(shape_previous[1], shape_previous[2]),
                                    strides=(shape_previous[1], shape_previous[2]), padding='SAME', name='maxpoolc3')
        # fc6
        fusion2_flat6 = tl.layers.FlattenLayer(fusion2_pool5, name='flatten2')
        fusion2_fc6 = tl.layers.DenseLayer(fusion2_flat6, n_units=512, act=tf.nn.elu, name='fcc2')
        fusion2_drop6 = tl.layers.DropoutLayer(fusion2_fc6, 0.35, is_fix=True, name='drop6')
        # fc7
        fusion2_fc7 = tl.layers.DenseLayer(fusion2_drop6, n_units=512, act=tf.nn.elu, name='fcc3')
        fusion2_drop7 = tl.layers.DropoutLayer(fusion2_fc7, 0.5, is_fix=True, name='drop7')

        # concatenate3
        fusion3_drop7 = tl.layers.ConcatLayer(layer=[left_drop7, right_drop7, fusion1_drop7, fusion2_drop7], concat_dim=-1, name='concat_layer3')
        # ## release shape limitation
        # shape_previous = fusion3_drop7.outputs.get_shape()
        # pool3 = tl.layers.MaxPool2d(fusion3_drop7, filter_size=(shape_previous[0], shape_previous[1]),
        #                             strides=(shape_previous[0], shape_previous[1]), padding='SAME', name='relative_maxpool3')
        # fusion3_drop7 = tl.layers.FlattenLayer(pool3, name='flat1')

        # fc8
        fusion3_fc8 = tl.layers.DenseLayer(fusion3_drop7, n_units=1024, name='fcc4')
        # fc9
        predictions = tl.layers.DenseLayer(fusion3_fc8, n_units=1, name='fcc5')
        if fe == 5:
            return predictions.outputs, right_conv5.outputs, left_conv5.outputs
        elif fe == 1:
            return predictions.outputs, right_conv1.outputs, left_conv1.outputs
        elif fe == 7:
            return predictions.outputs, fusion1_conv5.outputs, fusion2_conv5.outputs
        else:
            return predictions.outputs
##---------------Single Image SR---------------------##
w_init = tf.contrib.layers.xavier_initializer()
b_init = None  # tf.constant_initializer(value=0.0)
def SR_IQA(input, name = "SR_IQA", fe=1):
    with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):
        tl.layers.set_name_reuse()
        net_in = tl.layers.InputLayer(input, name='input/images')
        net_h0 = tl.layers.Conv2d(net_in, df_dim, (3, 3), (1, 1), act=tf.nn.leaky_relu,
                        padding='SAME', W_init=w_init, name='h0/c')
        net_h1 = tl.layers.Conv2d(net_h0, df_dim, (3, 3), (1, 1), act=tf.nn.leaky_relu,
                        padding='SAME', W_init=w_init, name='h1/c')
        sum1 = tl.layers.ElementwiseLayer([net_h0, net_h1], tf.add, name='sum1')
        pool1=tl.layers.MaxPool2d(sum1,filter_size=(2,2),strides=(2,2),name='maxpool1')
        net_h2 = tl.layers.Conv2d(pool1, df_dim*2, (3, 3), (1, 1), act=tf.nn.leaky_relu,
                        padding='SAME', W_init=w_init, name='h2/c')
        net_h3 = tl.layers.Conv2d(net_h2, df_dim*2, (3, 3), (1, 1), act=tf.nn.leaky_relu,
                        padding='SAME', W_init=w_init, name='h3/c')
        sum2 = tl.layers.ElementwiseLayer([net_h3, net_h2], tf.add, name='sum2')
        pool2=tl.layers.MaxPool2d(sum2,filter_size=(2,2),strides=(2,2),name='maxpool2')
        net_h4 = tl.layers.Conv2d(pool2, df_dim*4, (3, 3), (1, 1), act=tf.nn.leaky_relu,
                        padding='SAME', W_init=w_init, name='h4/c')
        net_h5 = tl.layers.Conv2d(net_h4, df_dim*4, (3, 3), (1, 1), act=tf.nn.leaky_relu,
                        padding='SAME', W_init=w_init, name='h5/c')
        sum3 = tl.layers.ElementwiseLayer([net_h4, net_h5], tf.add, name='sum3')
        #pool3= spp_layer(sum3,4)
        shape_previous = sum3.outputs.get_shape()
        pool3 = tl.layers.MaxPool2d(sum3, filter_size=(shape_previous[1],shape_previous[2]),strides=(shape_previous[1],shape_previous[2]),name='relative_maxpool3')

        flat1 = tl.layers.FlattenLayer(pool3, name='flat1')
        fc1 = tl.layers.DenseLayer(flat1, n_units=1024, name='fc1')
        fc2 = tl.layers.DenseLayer(fc1, n_units=1, name='fc2')

        if fe ==1:
            return fc2.outputs, sum3.outputs
        else:
            return fc2.outputs
def SRGAN_d(input_images, istrain=True, name="SRGAN_d", fs=16):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        x = lrelu(conv2d(input_images, fs, 3, 3, 4, 4, name='in'))
        f=x
        x = conv2d(x, fs, 3, 3, 2, 2, name='in0')
        x = lrelu(tf.layers.batch_normalization(x, training=istrain, name='bn1'))

        x = conv2d(x, fs*2, 3, 3, 2, 2, name='in1')
        x = lrelu(tf.layers.batch_normalization(x, training=istrain, name='bn2'))
        x = conv2d(x, fs * 2, 3, 3, 2, 2, name='in2')
        x = lrelu(tf.layers.batch_normalization(x, training=istrain, name='bn3'))

        x = conv2d(x, fs * 4, 3, 3, 2, 2, name='in3')
        x = lrelu(tf.layers.batch_normalization(x, training=istrain, name='bn4'))
        x = conv2d(x, fs * 4, 3,3, 2, 2, name='in4')
        x = lrelu(tf.layers.batch_normalization(x, training=istrain, name='bn5'))

        x = tf.layers.flatten(x)
        x = lrelu(tf.layers.dense(x, 64, tf.identity))
        x = tf.sigmoid(tf.layers.dense(x, 1, tf.identity))
    return x, f

def Vgg19_simple_api(rgb, reuse):
    VGG_MEAN = [103.939, 116.779, 123.68]
    with tf.variable_scope("VGG19", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        print("build model started")
        rgb_scaled = rgb * 255.0
        # Convert RGB to BGR
        if tf.__version__ <= '0.11':
            red, green, blue = tf.split(3, 3, rgb_scaled)
        else:  # TF 1.0
            # print(rgb_scaled)
            red, green, blue = tf.split(rgb_scaled, 3, 3)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        if tf.__version__ <= '0.11':
            bgr = tf.concat(3, [
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ])
        else:
            bgr = tf.concat(
                [
                    blue - VGG_MEAN[0],
                    green - VGG_MEAN[1],
                    red - VGG_MEAN[2],
                ], axis=3)
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        """ input layer """
        net_in = InputLayer(bgr, name='input')
        """ conv1 """
        network = Conv2d(net_in, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_1')
        network = Conv2d(network, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_2')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')
        """ conv2 """
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_1')
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_2')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2')
        """ conv3 """
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_1')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_2')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_3')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3')
        """ conv4 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool4')  # (batch_size, 14, 14, 512)
        conv = network
        """ conv5 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool5')  # (batch_size, 7, 7, 512)
        """ fc 6~8 """
        network = FlattenLayer(network, name='flatten')
        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc6')
        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc7')
        network = DenseLayer(network, n_units=1000, act=tf.identity, name='fc8')
        return network, conv.outputs

###  CVPR18/19=====================
def stereo_sr_cvpr2018(bic_image, bic_cbcr, pre_bic_series, name="stereo_sr_cvpr2018"):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        ## Y SR
        input = tf.concat([bic_image, pre_bic_series], 3)
        for i in range(0, 16):
            output = relu(conv2d(input, fm_n, 3, 3, name='conv'+str(i), init_value=init_value2))
            input=output
        residual_image=conv2d(input, 1, 3, 3, name='conv16', init_value=init_value2)
        high_res_luminance=tf.add(residual_image,bic_image)

        ## Color SR
        input_chrominance = tf.concat([high_res_luminance, bic_cbcr], 3)
        bic_image2 = tf.concat([bic_image, bic_cbcr], 3)
        input=input_chrominance
        for i in range(0,14):
            output=relu(conv2d(input, fm_n, 3, 3, name='conv'+str(17+i), init_value=init_value2))
            input=output
        residual_image2 = relu(conv2d(input, 3, 3, 3, name='conv31', init_value=init_value2))
        concatt = tf.concat([residual_image2, bic_image2],3)
        output = conv2d(concatt, 3, 3, 3, name='conv32', init_value=init_value2)
        return high_res_luminance, output

## ###============================== keras ==================================== ##
def definemodel():
    # left image
    left_image = Input(shape=(32, 32, 3))
    # conv1
    left_conv1 = Conv2D(32, (3, 3), padding='same', name='conv1_left')(left_image)
    left_elu1 = ELU()(left_conv1)
    left_pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1_left')(left_elu1)
    # conv2
    left_conv2 = Conv2D(32, (3, 3), padding='same', name='conv2_left')(left_pool1)
    left_elu2 = ELU()(left_conv2)
    left_pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2_left')(left_elu2)
    # conv3
    left_conv3 = Conv2D(64, (3, 3), padding='same', name='conv3_left')(left_pool2)
    left_elu3 = ELU()(left_conv3)
    # conv4
    left_conv4 = Conv2D(64, (3, 3), padding='same', name='conv4_left')(left_elu3)
    left_elu4 = ELU()(left_conv4)
    # conv5
    left_conv5 = Conv2D(128, (3, 3), padding='same', name='conv5_left')(left_elu4)
    left_elu5 = ELU()(left_conv5)
    left_pool5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool5_left')(left_elu5)
    # fc6
    left_flat6 = Flatten()(left_pool5)
    left_fc6 = Dense(512)(left_flat6)
    left_elu6 = ELU()(left_fc6)
    left_drop6 = Dropout(0.35)(left_elu6)
    # fc7
    left_fc7 = Dense(512)(left_drop6)
    left_elu7 = ELU()(left_fc7)
    left_drop7 = Dropout(0.5)(left_elu7)

    # right image
    right_image = Input(shape=(32, 32, 3))
    # conv1
    right_conv1 = Conv2D(32, (3, 3), padding='same', name='conv1_right')(right_image)
    right_elu1 = ELU()(right_conv1)
    right_pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1_right')(right_elu1)
    # conv2
    right_conv2 = Conv2D(32, (3, 3), padding='same', name='conv2_right')(right_pool1)
    right_elu2 = ELU()(right_conv2)
    right_pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2_right')(right_elu2)
    # conv3
    right_conv3 = Conv2D(64, (3, 3), padding='same', name='conv3_right')(right_pool2)
    right_elu3 = ELU()(right_conv3)
    # conv4
    right_conv4 = Conv2D(64, (3, 3), padding='same', name='conv4_right')(right_elu3)
    right_elu4 = ELU()(right_conv4)
    # conv5
    right_conv5 = Conv2D(128, (3, 3), padding='same', name='conv5_right')(right_elu4)
    right_elu5 = ELU()(right_conv5)
    right_pool5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool5_right')(right_elu5)
    # fc6
    right_flat6 = Flatten()(right_pool5)
    right_fc6 = Dense(512)(right_flat6)
    right_elu6 = ELU()(right_fc6)
    right_drop6 = Dropout(0.35)(right_elu6)
    # fc7
    right_fc7 = Dense(512)(right_drop6)
    right_elu7 = ELU()(right_fc7)
    right_drop7 = Dropout(0.5)(right_elu7)

    # concatenate1
    add_conv2 = keras.layers.add([left_conv2, right_conv2])
    subtract_conv2 = keras.layers.subtract([left_conv2, right_conv2])
    fusion1_conv2 = keras.layers.concatenate([add_conv2, subtract_conv2], axis=-1)
    #merge([add_conv2, subtract_conv2], mode='concat')
    fusion1_elu2 = ELU()(fusion1_conv2)
    fusion1_pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2_fusion1')(fusion1_elu2)
    # conv3
    fusion1_conv3 = Conv2D(64, (3, 3), padding='same', name='conv3_fusion1')(fusion1_pool2)
    fusion1_elu3 = ELU()(fusion1_conv3)
    # conv4
    fusion1_conv4 = Conv2D(64, (3, 3), padding='same', name='conv4_fusion1')(fusion1_elu3)
    fusion1_elu4 = ELU()(fusion1_conv4)
    # conv5
    fusion1_conv5 = Conv2D(128, (3, 3), padding='same', name='conv5_fusion1')(fusion1_elu4)
    fusion1_elu5 = ELU()(fusion1_conv5)
    fusion1_pool5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool5_fusion1')(fusion1_elu5)
    # fc6
    fusion1_flat6 = Flatten()(fusion1_pool5)
    fusion1_fc6 = Dense(512)(fusion1_flat6)
    fusion1_elu6 = ELU()(fusion1_fc6)
    fusion1_drop6 = Dropout(0.35)(fusion1_elu6)
    # fc7
    fusion1_fc7 = Dense(512)(fusion1_drop6)
    fusion1_elu7 = ELU()(fusion1_fc7)
    fusion1_drop7 = Dropout(0.5)(fusion1_elu7)

    # concatenate2
    add_conv5 = keras.layers.add([left_conv5, right_conv5])
    subtract_conv5 = keras.layers.subtract([left_conv5, right_conv5])
    fusion2_conv5 = keras.layers.concatenate([add_conv5, subtract_conv5], axis=-1)
    fusion2_elu5 = ELU()(fusion2_conv5)
    fusion2_pool5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2_fusion2')(fusion2_elu5)
    # fc6
    fusion2_flat6 = Flatten()(fusion2_pool5)
    fusion2_fc6 = Dense(512)(fusion2_flat6)
    fusion2_elu6 = ELU()(fusion2_fc6)
    fusion2_drop6 = Dropout(0.35)(fusion2_elu6)
    # fc7
    fusion2_fc7 = Dense(512)(fusion2_drop6)
    fusion2_elu7 = ELU()(fusion2_fc7)
    fusion2_drop7 = Dropout(0.5)(fusion2_elu7)

    # concatenate3
    fusion3_drop7 = keras.layers.concatenate([left_drop7, right_drop7, fusion1_drop7, fusion2_drop7], axis=-1)
    # fc8
    fusion3_fc8 = Dense(1024)(fusion3_drop7)
    # fc9
    predictions = Dense(1)(fusion3_fc8)
    return left_image, right_image, predictions
def definemodel_tf(left_image, right_image):
    # left image Input(shape=(32, 32, 3))
    # conv1
    left_conv1 = Conv2D(32, (3, 3), padding='same', name='conv1_left')(left_image)
    left_elu1 = ELU()(left_conv1)
    left_pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1_left')(left_elu1)
    # conv2
    left_conv2 = Conv2D(32, (3, 3), padding='same', name='conv2_left')(left_pool1)
    left_elu2 = ELU()(left_conv2)
    left_pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2_left')(left_elu2)
    # conv3
    left_conv3 = Conv2D(64, (3, 3), padding='same', name='conv3_left')(left_pool2)
    left_elu3 = ELU()(left_conv3)
    # conv4
    left_conv4 = Conv2D(64, (3, 3), padding='same', name='conv4_left')(left_elu3)
    left_elu4 = ELU()(left_conv4)
    # conv5
    left_conv5 = Conv2D(128, (3, 3), padding='same', name='conv5_left')(left_elu4)
    left_elu5 = ELU()(left_conv5)
    left_pool5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool5_left')(left_elu5)
    # fc6
    left_flat6 = Flatten()(left_pool5)
    left_fc6 = Dense(512)(left_flat6)
    left_elu6 = ELU()(left_fc6)
    left_drop6 = Dropout(0.35)(left_elu6)
    # fc7
    left_fc7 = Dense(512)(left_drop6)
    left_elu7 = ELU()(left_fc7)
    left_drop7 = Dropout(0.5)(left_elu7)

    # right image
    # conv1
    right_conv1 = Conv2D(32, (3, 3), padding='same', name='conv1_right')(right_image)
    right_elu1 = ELU()(right_conv1)
    right_pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1_right')(right_elu1)
    # conv2
    right_conv2 = Conv2D(32, (3, 3), padding='same', name='conv2_right')(right_pool1)
    right_elu2 = ELU()(right_conv2)
    right_pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2_right')(right_elu2)
    # conv3
    right_conv3 = Conv2D(64, (3, 3), padding='same', name='conv3_right')(right_pool2)
    right_elu3 = ELU()(right_conv3)
    # conv4
    right_conv4 = Conv2D(64, (3, 3), padding='same', name='conv4_right')(right_elu3)
    right_elu4 = ELU()(right_conv4)
    # conv5
    right_conv5 = Conv2D(128, (3, 3), padding='same', name='conv5_right')(right_elu4)
    right_elu5 = ELU()(right_conv5)
    right_pool5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool5_right')(right_elu5)
    # fc6
    right_flat6 = Flatten()(right_pool5)
    right_fc6 = Dense(512)(right_flat6)
    right_elu6 = ELU()(right_fc6)
    right_drop6 = Dropout(0.35)(right_elu6)
    # fc7
    right_fc7 = Dense(512)(right_drop6)
    right_elu7 = ELU()(right_fc7)
    right_drop7 = Dropout(0.5)(right_elu7)

    # concatenate1
    add_conv2 = keras.layers.add([left_conv2, right_conv2])
    subtract_conv2 = keras.layers.subtract([left_conv2, right_conv2])
    fusion1_conv2 = keras.layers.concatenate([add_conv2, subtract_conv2], axis=-1)
    #merge([add_conv2, subtract_conv2], mode='concat')
    fusion1_elu2 = ELU()(fusion1_conv2)
    fusion1_pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2_fusion1')(fusion1_elu2)
    # conv3
    fusion1_conv3 = Conv2D(64, (3, 3), padding='same', name='conv3_fusion1')(fusion1_pool2)
    fusion1_elu3 = ELU()(fusion1_conv3)
    # conv4
    fusion1_conv4 = Conv2D(64, (3, 3), padding='same', name='conv4_fusion1')(fusion1_elu3)
    fusion1_elu4 = ELU()(fusion1_conv4)
    # conv5
    fusion1_conv5 = Conv2D(128, (3, 3), padding='same', name='conv5_fusion1')(fusion1_elu4)
    fusion1_elu5 = ELU()(fusion1_conv5)
    fusion1_pool5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool5_fusion1')(fusion1_elu5)
    # fc6
    fusion1_flat6 = Flatten()(fusion1_pool5)
    fusion1_fc6 = Dense(512)(fusion1_flat6)
    fusion1_elu6 = ELU()(fusion1_fc6)
    fusion1_drop6 = Dropout(0.35)(fusion1_elu6)
    # fc7
    fusion1_fc7 = Dense(512)(fusion1_drop6)
    fusion1_elu7 = ELU()(fusion1_fc7)
    fusion1_drop7 = Dropout(0.5)(fusion1_elu7)

    # concatenate2
    add_conv5 = keras.layers.add([left_conv5, right_conv5])
    subtract_conv5 = keras.layers.subtract([left_conv5, right_conv5])
    fusion2_conv5 = keras.layers.concatenate([add_conv5, subtract_conv5], axis=-1)
    fusion2_elu5 = ELU()(fusion2_conv5)
    fusion2_pool5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2_fusion2')(fusion2_elu5)
    # fc6
    fusion2_flat6 = Flatten()(fusion2_pool5)
    fusion2_fc6 = Dense(512)(fusion2_flat6)
    fusion2_elu6 = ELU()(fusion2_fc6)
    fusion2_drop6 = Dropout(0.35)(fusion2_elu6)
    # fc7
    fusion2_fc7 = Dense(512)(fusion2_drop6)
    fusion2_elu7 = ELU()(fusion2_fc7)
    fusion2_drop7 = Dropout(0.5)(fusion2_elu7)

    # concatenate3
    fusion3_drop7 = keras.layers.concatenate([left_drop7, right_drop7, fusion1_drop7, fusion2_drop7], axis=-1)
    # fc8
    fusion3_fc8 = Dense(1024)(fusion3_drop7)
    # fc9
    predictions = Dense(1)(fusion3_fc8)
    return predictions

def SR_localbic_keras(left_image, fsize=128,scale = 4,small=True):
    with tf.variable_scope("pret_ed",reuse =tf.AUTO_REUSE) as scope:
        # conv1
        x = Conv2D(fsize, (3, 3), padding='same', name='in')(left_image)
        conv_1 = x
        for i in range(6):
            x = ResBlock_edsr(x, name='res' + str(i), f_size=fsize)
        x = Conv2D(fsize, (3, 3), padding='same', name='conv1')(x)
        x += conv_1

        x = Conv2D(fsize, (3, 3), padding='same', name='convim')(x)
        x = LeakyReLU()(x)

        x = Conv2D(fsize, (3, 3), padding='same', name='convf')(x)
        x1 = LeakyReLU()(x)

        x = Conv2D(fsize, (3, 3), padding='same', name='convf1')(x)
        x2 = LeakyReLU()(x)

        f = Concatenate()([conv_1, x1, x2])
        f = Conv2D(fsize, (3, 3), padding='same', name='convout0')(f)

        # Upsample output of the convolution
        x = Conv2DTranspose(fsize, 8, (scale, scale), 'same', name='deconv')(f)
        output = Conv2D(3, (3, 3), padding='same', name='out')(x)

        if small:
            return f, output
        else:
            return x, output
