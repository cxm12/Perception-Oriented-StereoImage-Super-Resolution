import argparse
import os
from modelall import *
from utills import *
from PIL import Image
from skimage import measure
from scipy import misc


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
method = 'PSSR'
isFR = 0
contrastive = False  # True  #
parser = argparse.ArgumentParser(description=method)
parser.add_argument('--batch', default=12)
parser.add_argument('--patch', default=200)
parser.add_argument('--scale', default=4)
args = parser.parse_args()
modelpath = './Checkpoint/SR/s' + str(args.scale) + '/' + method + '/'
if not os.path.exists(modelpath):
    os.makedirs(modelpath)
    
iqastereopath = './Checkpoint/IQA/BDTL_res_all_10/model.ckpt-454000'  # './Checkpoint/IQATLresall_10/model.ckpt-2855000'
nriqastereopath = './Checkpoint/IQA/BDTL_all_10/model.ckpt-1294000'

#####===================TEST==========================######
testpath = './testset/left/'
visualization = True  # False#


def testSR():
    save = True
    filenames = get_filenames([testpath])
    num = len(filenames)  #
    
    LR = tf.convert_to_tensor(tf.placeholder(tf.float32, [1, args.patch // args.scale, args.patch // args.scale, 3]))
    fl, sl = SR_localbic(LR, tf.image.resize_bicubic(LR, [args.patch, args.patch]), 128, scale=args.scale,
                         small=False)
    res, resr = SR_ctxLeftRight(sl, sl, fl, fl, s=args.scale, small=False)
    
    s = modelIQATL(res, resr, name='IQA', fe=0)
    
    if save:
        savepath = './result/s' + str(args.scale) + '/' + method + '/right/'
        os.makedirs(savepath, exist_ok=True)
    mean_score = 0
    mean_psnr = 0
    mean_ssim = 0
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        with tf.variable_scope('', reuse=True):
            sess.run(init)
            model_vars = tf.trainable_variables()  #
            iqavar = [var for var in model_vars if 'IQA' in var.name]
            srvar = [var for var in model_vars if 'pret_ed' in var.name]
            srvar1 = [var for var in model_vars if 'context' in var.name]
        
            saver = tf.train.Saver(var_list=srvar + srvar1, max_to_keep=500)
            saveriqa = tf.train.Saver(var_list=iqavar, max_to_keep=500)
            if isFR:
                saveriqa.restore(sess, iqastereopath)
            else:
                saveriqa.restore(sess, nriqastereopath)
        
            module_file = modelpath + 'model.ckpt-192000'
            saver.restore(sess, module_file)
            for fi in range(0, num):
                fname = filenames[fi][len(testpath):-4].replace('left', 'right')
                leftgt = np.array(Image.open(filenames[fi]))
                rightgt = np.array(Image.open(filenames[fi].replace('left', 'right').replace('_L', '_R')))
                h, w, _ = leftgt.shape
                h = h // args.scale * args.scale
                w = w // args.scale * args.scale
                leftgt = leftgt[:h, :w, :3]
                rightgt = rightgt[:h, :w, :3]
                h, w, c0 = leftgt.shape
            
                left = misc.imresize(leftgt, 1 / args.scale, 'bicubic')[:, :, :3]
                leftb = misc.imresize(left, [h, w, c0], 'bicubic')
                right = misc.imresize(rightgt, 1 / args.scale, 'bicubic')[:, :, :3]
                rightb = misc.imresize(rightgt, [h, w, c0], 'bicubic')
            
                left = np.reshape(np.float32(left / 255), [1, h // args.scale, w // args.scale, c0])
                leftb = np.reshape(np.float32(leftb / 255), [1, h, w, c0])
                right = np.reshape(np.float32(right / 255), [1, h // args.scale, w // args.scale, c0])
                rightb = np.reshape(np.float32(rightb / 255), [1, h, w, c0])
                leftgt = np.reshape(np.float32(leftgt / 255), [1, h, w, c0])
                rightgt = np.reshape(np.float32(rightgt / 255), [1, h, w, c0])
                leftgt1 = tf.reshape(
                    tf.extract_image_patches(leftgt, (1, 120, 120, 1), (1, 120, 120, 1), (1, 1, 1, 1), 'VALID')
                    , [-1, 120, 120, 3])
                rightgt1 = tf.reshape(
                    tf.extract_image_patches(rightgt, (1, 120, 120, 1), (1, 120, 120, 1), (1, 1, 1, 1), 'VALID')
                    , [-1, 120, 120, 3])
                
                fl, sl = SR_localbic(tf.convert_to_tensor(left), tf.convert_to_tensor(leftb), 128,
                                     scale=args.scale, small=False)
                fr, sr = SR_localbic(tf.convert_to_tensor(right), tf.convert_to_tensor(rightb), 128,
                                     scale=args.scale, small=False)
                out, outr = SR_ctxLeftRight(sl, sr, fl, fr, s=args.scale, small=False)
                
            
                out1 = tf.extract_image_patches(out, (1, 120, 120, 1), (1, 120, 120, 1), (1, 1, 1, 1), 'VALID')
                out2 = tf.reshape(out1, [-1, 120, 120, 3])
                outr1 = tf.extract_image_patches(outr, (1, 120, 120, 1), (1, 120, 120, 1), (1, 1, 1, 1), 'VALID')
                outr2 = tf.reshape(outr1, [-1, 120, 120, 3])
                if isFR:
                    s = modelIQATL(leftgt1 - out2, rightgt1 - outr2, name='IQA', fe=0)
                else:
                    s = modelIQATL(out2, outr2, name='IQA', fe=0)
            
                score = sess.run(s)  # print(score.shape)
                score = np.mean(score)
                result = np.reshape(score, [1])
                print('image ', fi, 'score = ', result)
                mean_score += result
            
                if True:
                    res0, resr0 = sess.run([out, outr])
                    result = np.reshape(res0, [h, w, c0])
                    resultr = np.reshape(resr0, [h, w, c0])
                    rgbresult1 = np.maximum(0, np.minimum(1, np.reshape(result, [h, w, c0])))
                    if save:
                        Image.fromarray(img_to_uint8(
                            np.round(np.maximum(0, np.minimum(1, np.reshape(resultr, [h, w, c0]))) * 255))).save(
                            savepath + fname + '.png')
                    c = np.float32(255 * leftgt[0, args.scale:-args.scale, args.scale:-args.scale, :])
                    sr = rgbresult1[args.scale:-args.scale, args.scale:-args.scale, :] * 255
                    psnr1, _ = psnr(c, sr)
                    mean_psnr += psnr1
                    score = measure.compare_ssim(c / 255.0, sr / 255.0, multichannel=True)  #
                    mean_ssim += score
                    print('RGB:', psnr1, score)
            mean_score /= num
            mean_psnr = mean_psnr / num
            mean_ssim = mean_ssim / num
            print('IQA result:', mean_score, 'result:', mean_psnr, 'SSIM', mean_ssim)


if __name__ == '__main__':
    testSR()
