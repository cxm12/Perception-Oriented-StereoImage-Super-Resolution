# higher predicted quality values represent lower visual quality
from modelall import *
import argparse
import os
from utills import *
import glob
from PIL import Image


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 'StereoSRIQA'
parser = argparse.ArgumentParser(description='IQA')
parser.add_argument('--batch', default=32)
parser.add_argument('--scale', default=4)
args = parser.parse_args()


#############======================  Test Stereo IQA ==========================  ##
def dive_patch(img, p=32, steps=64):
    h, w, _ = img.shape
    patch = []
    for i in range(0, h - p + 1, steps):  #
        for j in range(0, w - p + 1, steps):
            patch.append(img[i: i + p, j:j + p, :])
    p = np.array(patch)  # [80, 32, 32, 3]
    return p


def get_filenames(paths):  # no sub fileset
    filenames = []
    for path in paths:
        f = os.walk(path)
        root, dirs, files = 0, 0, 0
        for root, dirs, files in f:
            # print([root, dirs, files])
            break
        if files == 0:
            continue
        for fil in files:
            filenames.append(os.path.join(root, fil))
    return filenames


isFR = 0
method = 'BDTL_all_10'


def test():
    testpath = "./result/s4/left/"
    filenames = glob.glob(testpath + "/*.png") + glob.glob(testpath + "/*.bmp")
    print(filenames)
    pw = 80
    num = len(filenames)  # 3 #
    lr = tf.convert_to_tensor(tf.placeholder(tf.float32, [1, pw, pw, 3]))
    s = modelIQATL(lr, lr, name='IQA')

    init = tf.global_variables_initializer()
    s = 0
    with tf.Session() as sess:
        with tf.variable_scope('', reuse=True):
            sess.run(init)
            model_vars = tf.trainable_variables()  #
            saver = tf.train.Saver(var_list=model_vars, max_to_keep=10000)
        
            module_file = './Checkpoint/IQA/' + method + '/' + 'model.ckpt-1294000'
            saver.restore(sess, module_file)
        
            scorelst = []
            for fi in range(0, num):
                fname = filenames[fi][len(testpath):]
                left = np.array(Image.open(filenames[fi]))
                right = np.array(Image.open(filenames[fi].replace('left', 'right').replace('_L', '_R')))
                if isFR:
                    leftgt = np.array(Image.open("./testset/left/" + fname + '_left.bmp'))  # _right
                    rightgt = np.array(Image.open(("./testset/left/" + fname + '_left.bmp').replace('left', 'right')))
                else:
                    leftgt = left
                    rightgt = right
                h, w, c = left.shape
                leftgt = leftgt[:h, :w, :3]
                left = left[:h, :w, :3]
                rightgt = rightgt[:h, :w, :3]
                right = right[:h, :w, :3]
            
                left = np.float32(left / 255)
                leftgt = np.float32(leftgt / 255)
                right = np.float32(right / 255)
                rightgt = np.float32(rightgt / 255)
                cur_diffbc = leftgt - left
                cur_diffbcr = rightgt - right
                df = []
                dfr = []
                lgt = []
                l = []
                rgt = []
                r = []
                count = 0
                for row in range(0, h - pw + 1, pw // 2):
                    for col in range(0, w - pw + 1, pw // 2):
                        count = count + 1
                        cur_pbc = cur_diffbc[row:row + pw, col:col + pw]
                        cur_pbcr = cur_diffbcr[row:row + pw, col:col + pw]
                        df.append(cur_pbc)
                        dfr.append(cur_pbcr)
                        l.append(left[row:row + pw, col:col + pw])
                        lgt.append(leftgt[row:row + pw, col:col + pw])
                        rgt.append(rightgt[row:row + pw, col:col + pw])
                        r.append(right[row:row + pw, col:col + pw])
                r = np.reshape(np.array(r), [count, pw, pw, 3])
                rgt = np.reshape(np.array(rgt), [count, pw, pw, 3])
                lgt = np.reshape(np.array(lgt), [count, pw, pw, 3])
                l = np.reshape(np.array(l), [count, pw, pw, 3])
            
                if isFR:
                    out = modelIQATL(tf.convert_to_tensor(lgt) - tf.convert_to_tensor(l),
                                     tf.convert_to_tensor(rgt) - tf.convert_to_tensor(r), name='IQA')
                else:
                    out = modelIQATL(tf.convert_to_tensor(l), tf.convert_to_tensor(r), name='IQA')
            
                result = np.reshape(sess.run(out), [count])
                scorelst.append(np.mean(result))
                s += np.mean(result)
                print('score = ', np.mean(result))
    s /= num
    print('test', testpath, 'all images\' scores', scorelst, '\t average score =', s)


if __name__ == '__main__':
    test()
