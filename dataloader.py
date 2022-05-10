from scipy import misc
from test_Middlebury import get_filenames
import os
from utills import *
from utills import gaussian_kernel_2d_opencv
import random

####### ========= imgall dataset ============ ##
class DataLoaderIQA():
    def __init__(self, opt):
        self.data_dirgt = opt.dataimgt
        self.data_dir = opt.dataim
        self.data_score = opt.datascore
        self.filename = get_filenames([self.data_dir + 'left/'])
        random.shuffle(self.filename)
        # self.methodname = ['GT'], 'GT',
        self.methodname = ['quarter_cvpr20', 'full_cvpr18', 'full_cvpr19', 'bicubic', 'full_cvpr20']
        self.scalelst = [2, 3, 4, 8, 16]
        self.patch_size = opt.patch
        # self.scale = opt.scale  # SR scale
        self.batch_size = opt.batch
        self.shuffle_num = 200
        self.prefetch_num = 100
        self.map_parallel_num = 8
    def get_RGBp(self):
        p = self.patch_size
        steps = p // 2
        filenamesgt = get_filenames([self.data_dirgt+'left/'])
        for methods in self.methodname:
            # methods = random.choice(self.methodname)
            if methods != 'GT':
                for scale in self.scalelst:
                    impath = [self.data_dir + methods + '/left/s' + str(scale) + '/']
                    impathr = [self.data_dir + methods + '/right/s' + str(scale) + '/']
                    # print(impath)
                    scorepath = [self.data_score + 's' + str(scale) + '_' + methods + '.mat']
                    filename = get_filenames(impath)
                    # print('=====================filenames ========================= ', filename)
                    # filenamer = get_filenames(impathr)
                    if len(filename) != 0:
                        print('methods:', methods, 'scale:', scale)
                        mat = sio.loadmat(scorepath[0])
                        score = mat['score']
                        psnr = mat['psnr']
                        ssim = mat['ssim']
                        names = mat['name']
                        salllst = mat['all']
                        for i in range(len(filename)):
                            name = names[i].replace(" ", "")  # filename[i][len(impath[0]):]
                            # print('name = ', name, 'length', len(name))
                            Left = np.array(misc.imread(impath[0]+name))  # filename[i]))
                            Right = misc.imread(impathr[0] + name)  #np.array(misc.imread(filename[i].replace('left', 'right')))  #
                            # if name[:-9] in filenamer[i]:
                            #     Right = np.array(misc.imread(impathr[0]+names[i]))  # filename[i].replace('left', 'right')))
                            # else:
                            #     print('Right/Left image name = ', filenamer[i], filename[i])
                            #     Right = np.array(misc.imread(filenamer[i]))
                            # if not name[:-9] in filenamesgt[i]:
                            #     print('GT image/SR image name = ', filenamesgt[i], filename[i])
                            Leftgt = np.array(misc.imread(self.data_dirgt+'left/'+name)) #filenamesgt[i]))
                            Rightgt = np.array(misc.imread(self.data_dirgt+'right/'+name)) #filenamesgt[i].replace('left', 'right')))  #
                            h, w, c = Leftgt.shape

                            # print('list(names) =================', list(names))
                            id = list(names).index(names[i])
                            # if not name[:-9] in names[i]:
                            #     print('name in mat is not == name in image file', names[i], name)
                            s = score[0][id]
                            ps = psnr[0][id]
                            ss = ssim[0][id]
                            sall = (salllst[0][id]) * 10  # min(max(lamda * (0.5 * (mxs / s) + 0.3 * (ps / mxps) + 0.2 * (ss / mxss)), 0), 1)
                            # print(name, sall, id, name, s, ps, ss)

                            imgrgbR = Right
                            imgrgb = Left
                            if len(imgrgb.shape) >= 3:
                                imgrgb = imgrgb[:, :, :3]
                                imgrgbR = imgrgbR[:, :, :3]
                            else:
                                imgrgb = np.expand_dims(imgrgb, -1)
                                imgrgbR = np.expand_dims(imgrgbR, -1)
                                imgrgb = np.concatenate((imgrgb, imgrgb, imgrgb), -1)
                                imgrgbR = np.concatenate((imgrgbR, imgrgbR, imgrgbR), -1)

                            imgor = misc.imresize(imgrgb, [h, w, c], 'bicubic')
                            imgorR = misc.imresize(imgrgbR, [h, w, c], 'bicubic')

                            for fd in range(0, 6):
                                if (fd == 0):
                                    imggt = Leftgt
                                    imggtR = Rightgt
                                    img = imgor
                                    imgR = imgorR
                                elif (fd == 1):
                                    imggt = np.rot90(Leftgt, 1)
                                    imggtR = np.rot90(Rightgt, 1)
                                    img = np.rot90(imgor, 1)
                                    imgR = np.rot90(imgorR, 1)
                                elif (fd == 2):
                                    imggt = np.rot90(Leftgt, 2)
                                    imggtR = np.rot90(Rightgt, 2)
                                    img = np.rot90(imgor, 2)
                                    imgR = np.rot90(imgorR, 2)
                                elif (fd == 3):
                                    imggt = np.rot90(Leftgt, 3)
                                    imggtR = np.rot90(Rightgt, 3)
                                    img = np.rot90(imgor, 3)
                                    imgR = np.rot90(imgorR, 3)
                                elif (fd == 4):
                                    imggt = np.flip(Leftgt, 1)
                                    imggtR = np.flip(Rightgt, 1)
                                    img = np.flip(imgor, 1)
                                    imgR = np.flip(imgorR, 1)
                                else:
                                    imggt = np.flip(Leftgt, 0)
                                    imggtR = np.flip(Rightgt, 0)
                                    img = np.flip(imgor, 0)
                                    imgR = np.flip(imgorR, 0)

                                h, w, c = img.shape
                                imglgt = imggt
                                imgRlgt = imggtR
                                imgl = img
                                imgRl = imgR

                                for i in range(0, h - p + 1, steps):  #
                                    for j in range(0, w - p + 1, steps):
                                        HRgt = imglgt[i: i + p, j:j + p, :]
                                        HRRgt = imgRlgt[i: i + p, j:j + p, :]
                                        HR = imgl[i: i + p, j:j + p, :]
                                        HRR = imgRl[i: i + p, j:j + p, :]

                                        HR = np.float32(HR / 255.0)
                                        HRgt = np.float32(HRgt / 255.0)
                                        HRR = np.float32(HRR / 255.0)
                                        HRRgt = np.float32(HRRgt / 255.0)
                                        c1 = np.random.rand()
                                        c2 = np.random.rand()
                                        if c1 < 0.5:
                                            HR, HRR = HR[::-1, :, :], HRR[::-1, :, :]
                                            HRgt, HRRgt = HRgt[::-1, :, :], HRRgt[::-1, :, :]
                                        if c2 < 0.5:
                                            HR, HRR = HR[:, ::-1, :], HRR[:, ::-1, :]
                                            HRgt, HRRgt = HRgt[:, ::-1, :], HRRgt[:, ::-1, :]
                                        yield HR, HRR, HRgt, HRRgt, sall
            else:
                scorepath = [self.data_score + methods + '.mat']
                if len(filenamesgt) != 0:
                    print('methods:', methods)
                    mat = sio.loadmat(scorepath[0])
                    score = mat['score']
                    psnr = mat['psnr']
                    ssim = mat['ssim']
                    names = mat['name']
                    salllst = mat['all']
                    # print('name in mat: ', names)
                    for i in range(len(filenamesgt)):
                        name = names[i].replace(" ", "")  # filenamesgt[i][len(self.data_dirgt+'left/'):]
                        Leftgt = np.array(misc.imread(self.data_dirgt+'left/'+name))  #filenamesgt[i]))
                        Left = Leftgt
                        Rightgt = np.array(misc.imread(self.data_dirgt+'right/'+name))  #filenamesgt[i].replace('left', 'right')))  #
                        Right = Rightgt
                        h, w, c = Leftgt.shape

                        id = list(names).index(names[i])
                        # if not name[:-9] in names[i]:
                        #     print('name in mat is not == name in image file', names[i], name)
                        s = score[0][id]
                        ps = psnr[0][id]
                        ss = ssim[0][id]
                        sall = (salllst[0][id])*10
                        # print(name, sall, id, name, s, ps, ss)

                        imgrgbR = Right
                        imgrgb = Left
                        if len(imgrgb.shape) >= 3:
                            imgrgb = imgrgb[:, :, :3]
                            imgrgbR = imgrgbR[:, :, :3]
                        imgor = misc.imresize(imgrgb, [h, w, c], 'bicubic')
                        imgorR = misc.imresize(imgrgbR, [h, w, c], 'bicubic')

                        for fd in range(0, 6):
                            if (fd == 0):
                                imggt = Leftgt
                                imggtR = Rightgt
                                img = imgor
                                imgR = imgorR
                            elif (fd == 1):
                                imggt = np.rot90(Leftgt, 1)
                                imggtR = np.rot90(Rightgt, 1)
                                img = np.rot90(imgor, 1)
                                imgR = np.rot90(imgorR, 1)
                            elif (fd == 2):
                                imggt = np.rot90(Leftgt, 2)
                                imggtR = np.rot90(Rightgt, 2)
                                img = np.rot90(imgor, 2)
                                imgR = np.rot90(imgorR, 2)
                            elif (fd == 3):
                                imggt = np.rot90(Leftgt, 3)
                                imggtR = np.rot90(Rightgt, 3)
                                img = np.rot90(imgor, 3)
                                imgR = np.rot90(imgorR, 3)
                            elif (fd == 4):
                                imggt = np.flip(Leftgt, 1)
                                imggtR = np.flip(Rightgt, 1)
                                img = np.flip(imgor, 1)
                                imgR = np.flip(imgorR, 1)
                            else:
                                imggt = np.flip(Leftgt, 0)
                                imggtR = np.flip(Rightgt, 0)
                                img = np.flip(imgor, 0)
                                imgR = np.flip(imgorR, 0)

                            h, w, c = img.shape
                            imglgt = imggt
                            imgRlgt = imggtR
                            imgl = img
                            imgRl = imgR

                            for i in range(0, h - p + 1, steps):  #
                                for j in range(0, w - p + 1, steps):
                                    HRgt = imglgt[i: i + p, j:j + p, :]
                                    HRRgt = imgRlgt[i: i + p, j:j + p, :]
                                    HR = imgl[i: i + p, j:j + p, :]
                                    HRR = imgRl[i: i + p, j:j + p, :]

                                    HR = np.float32(HR / 255.0)
                                    HRgt = np.float32(HRgt / 255.0)
                                    HRR = np.float32(HRR / 255.0)
                                    HRRgt = np.float32(HRRgt / 255.0)
                                    c1 = np.random.rand()
                                    c2 = np.random.rand()
                                    if c1 < 0.5:
                                        HR, HRR = HR[::-1, :, :], HRR[::-1, :, :]
                                        HRgt, HRRgt = HRgt[::-1, :, :], HRRgt[::-1, :, :]
                                    if c2 < 0.5:
                                        HR, HRR = HR[:, ::-1, :], HRR[:, ::-1, :]
                                        HRgt, HRRgt = HRgt[:, ::-1, :], HRRgt[:, ::-1, :]
                                    yield HR, HRR, HRgt, HRRgt, sall

    ## all image in one fold:
    def get_RGBallp(self):
        p = self.patch_size
        steps = p // 2
        mat = sio.loadmat(self.data_score+'iqaall.mat')
        names = mat['name']
        salllst = mat['all']
        filename = self.filename
        for i in range(len(filename)):
            namemat = names[i].replace(" ", "")
            name = filename[i][len(self.data_dir+'left/'):]  # name = 'quarter_cvpr20s2_wood1.png'
            st = name.find('_')
            st1 = name[st+1:].find('_')
            if st1 == -1:  # 'GT'
                Left = np.array(misc.imread(self.data_dir+'left/'+name))#filename[i]))  #
                Right = misc.imread(self.data_dir+'right/'+name)#filename[i].replace('left', 'right'))  #
                namegt = name[st+1:]
                # print('name = ', name, 'namemat = ', namemat, 'namegt = ', namegt)
                Leftgt = np.array(misc.imread(self.data_dirgt + 'left/' + namegt))
                Rightgt = np.array(misc.imread(self.data_dirgt + 'right/' + namegt))
                # print('left', self.data_dir + 'left/' + name, '\t right', self.data_dir+'right/'+name)# filename[i].replace('left', 'right'))
                h, w, c = Leftgt.shape
            else:
                Left = np.array(misc.imread(self.data_dir + 'left/' + name))  # filename[i]))  #
                Right = misc.imread(self.data_dir + 'right/' + name)  # filename[i].replace('left', 'right'))  #
                # print('st = ', st, st1)
                namegt = name[st + st1 + 2:]
                # print('name = ', name, 'namemat = ', namemat, 'namegt = ', namegt)
                Leftgt = np.array(misc.imread(self.data_dirgt + 'left/' + namegt))  # filename[i]))
                Rightgt = np.array(misc.imread(self.data_dirgt + 'right/' + namegt))
                # print('left', self.data_dir + 'left/' + name, '\t right', self.data_dir + 'right/' + name)  # filename[i].replace('left', 'right'))
                h, w, c = Leftgt.shape
            nana = name
            for ni in range(31-len(name)):
                nana += ' '
            id = list(names).index(nana)
            print('name', name, ' in mat = ', str(id), 'name in mat = ', nana, 'names[id]', names[id])
            sall = (salllst[0][id])  * 10
            imgrgbR = Right
            imgrgb = Left
            if len(imgrgb.shape) >= 3:
                imgrgb = imgrgb[:, :, :3]
                imgrgbR = imgrgbR[:, :, :3]
            else:
                imgrgb = np.expand_dims(imgrgb, -1)
                imgrgbR = np.expand_dims(imgrgbR, -1)
                imgrgb = np.concatenate((imgrgb, imgrgb, imgrgb), -1)
                imgrgbR = np.concatenate((imgrgbR, imgrgbR, imgrgbR), -1)

            imgor = misc.imresize(imgrgb, [h, w, c], 'bicubic')
            imgorR = misc.imresize(imgrgbR, [h, w, c], 'bicubic')

            for fd in range(0, 6):
                if (fd == 0):
                    imggt = Leftgt
                    imggtR = Rightgt
                    img = imgor
                    imgR = imgorR
                elif (fd == 1):
                    imggt = np.rot90(Leftgt, 1)
                    imggtR = np.rot90(Rightgt, 1)
                    img = np.rot90(imgor, 1)
                    imgR = np.rot90(imgorR, 1)
                elif (fd == 2):
                    imggt = np.rot90(Leftgt, 2)
                    imggtR = np.rot90(Rightgt, 2)
                    img = np.rot90(imgor, 2)
                    imgR = np.rot90(imgorR, 2)
                elif (fd == 3):
                    imggt = np.rot90(Leftgt, 3)
                    imggtR = np.rot90(Rightgt, 3)
                    img = np.rot90(imgor, 3)
                    imgR = np.rot90(imgorR, 3)
                elif (fd == 4):
                    imggt = np.flip(Leftgt, 1)
                    imggtR = np.flip(Rightgt, 1)
                    img = np.flip(imgor, 1)
                    imgR = np.flip(imgorR, 1)
                else:
                    imggt = np.flip(Leftgt, 0)
                    imggtR = np.flip(Rightgt, 0)
                    img = np.flip(imgor, 0)
                    imgR = np.flip(imgorR, 0)

                h, w, c = img.shape
                imglgt = imggt
                imgRlgt = imggtR
                imgl = img
                imgRl = imgR

                for i in range(0, h - p + 1, steps):  #
                    for j in range(0, w - p + 1, steps):
                        HRgt = imglgt[i: i + p, j:j + p, :]
                        HRRgt = imgRlgt[i: i + p, j:j + p, :]
                        HR = imgl[i: i + p, j:j + p, :]
                        HRR = imgRl[i: i + p, j:j + p, :]

                        HR = np.float32(HR / 255.0)
                        HRgt = np.float32(HRgt / 255.0)
                        HRR = np.float32(HRR / 255.0)
                        HRRgt = np.float32(HRRgt / 255.0)
                        c1 = np.random.rand()
                        c2 = np.random.rand()
                        if c1 < 0.5:
                            HR, HRR = HR[::-1, :, :], HRR[::-1, :, :]
                            HRgt, HRRgt = HRgt[::-1, :, :], HRRgt[::-1, :, :]
                        if c2 < 0.5:
                            HR, HRR = HR[:, ::-1, :], HRR[:, ::-1, :]
                            HRgt, HRRgt = HRgt[:, ::-1, :], HRRgt[:, ::-1, :]
                        yield HR, HRR, HRgt, HRRgt, sall

    ## from PSNR, stereoIQA, SSIM to score
    def calculateScore(self):
        mxs = 60
        mxps = 50
        mxss = 1.0
        lamda = 0.45
        num = 0
        # savepathgt = './data/iqascore/GTall.mat'
        # sio.savemat(savepathgt, {'score': sc, 'name': na, 'psnr': 50 * np.ones([66]), 'ssim': np.ones([66]), 'all': np.ones([66])})

        for methods in self.methodname:
            for scale in self.scalelst:
                scorepath = [self.data_score + 's' + str(scale) + '_' + methods + '.mat']
                savepath = './data/iqascore/' + 's' + str(scale) + '_' + methods + 'all.mat'
                if os.path.exists(scorepath[0]):
                    num += 1
                    print('methods:', methods, 'scale:', scale)
                    mat = sio.loadmat(scorepath[0])
                    score = mat['score']
                    psnr = mat['psnr']
                    ssim = mat['ssim']
                    names = mat['name']
                    sall = []
                    for name in names:
                        i = list(names).index(name)
                        s = score[0][i]
                        ps = psnr[0][i]
                        ss = ssim[0][i]
                        sall0 = min(max(lamda*(0.5 * (mxs / s) + 0.3 * (ps / mxps) + 0.2 * (ss / mxss)), 0), 1)
                        sall.append(sall0)
                        # print(name, sall0, i, name, s, ps, ss)
                    sc = score[0]  # list(score[0])
                    na = names  #list(names)
                    pss = psnr[0]  # list(psnr[0])
                    ssi = ssim[0]  # list(ssim[0])
                    sa = np.array(sall)

                    # sio.savemat(savepath, {'score': score[0], 'name': names, 'psnr': psnr[0],
                    #             'ssim': ssim[0], 'all': sa})
                    sio.savemat(savepath, {'score': sc, 'name': na, 'psnr': pss, 'ssim': ssi, 'all': sa})
                    print('score = ', np.mean(sall))
                    np.savetxt(savepath[:-4] + ".txt", sall, fmt='%f', delimiter=',')
        print(num)
    def read_png(self):
        dataset = tf.data.Dataset.from_generator(self.get_RGBallp, (tf.float32, tf.float32,tf.float32, tf.float32,tf.float32))
        dataset = dataset.shuffle(self.shuffle_num).prefetch(self.prefetch_num)
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size)).repeat()
        gts, gtsr, im, imr, sc = dataset.make_one_shot_iterator().get_next()
        p = self.patch_size

        imr = tf.reshape(imr, [self.batch_size, p, p, 3])
        im = tf.reshape(im, [self.batch_size, p, p, 3])
        gts = tf.reshape(gts, [self.batch_size, p, p, 3])
        gtsr = tf.reshape(gtsr, [self.batch_size, p, p, 3])
        sc = tf.reshape(sc, [self.batch_size, 1])
        return gts, gtsr, im, imr, sc

##### imgblur ##############
class DataLoaderIQAnew():
    def __init__(self, opt):
        self.data_dirgt = opt.dataimgt
        self.data_dir = './data/imgblurall/'
        self.data_score = './data/imgblurall/iqa/leftiqa_all_scorelst.mat'
        self.filenamegt = get_filenames([self.data_dirgt + 'left/'])
        self.filename = get_filenames([self.data_dir + 'left/'])
        random.shuffle(self.filename)
        self.patch_size = opt.patch
        self.batch_size = opt.batch
        self.shuffle_num = 200
        self.prefetch_num = 100
        self.map_parallel_num = 8

    ## all image in one fold:  no GT images
    def get_RGBallp(self):
        norm = 1 # rescale score
        p = self.patch_size
        steps = p // 2
        mat = sio.loadmat(self.data_score)
        names = mat['name']
        length = len(names[0])
        rank = mat['rank']
        filename = self.filename
        for i in range(len(filename)):
            # namemat = names[i].replace(" ", "")  # bics2_Adiron.png  srcnns2_Adiron.png
            name = filename[i][len(self.data_dir+'left/'):]
            st = name.find('_')
            st1 = name[st+1:].find('_')
            Left = np.array(misc.imread(self.data_dir + 'left/' + name))
            Right = misc.imread(self.data_dir + 'right/' + name)
            namegt = name[st + st1 + 2:]
            # print('name = ', name, 'namegt = ', namegt)  # print('name = ', name, 'namemat = ', namemat, 'namegt = ', namegt)
            Leftgt = np.array(misc.imread(self.data_dirgt + 'left/' + namegt))  # filename[i]))
            Rightgt = np.array(misc.imread(self.data_dirgt + 'right/' + namegt))
            h, w, c = Leftgt.shape

            nana = name
            for ni in range(length-len(name)):
                nana += ' '
            id = list(names).index(nana)
            sall = (rank[0][id])  * norm
            # print('name in filename[i]=', name, ' in mat = ', str(id), 'name in mat names[id]=', names[id], 'rank[id]=', sall)
            imgrgbR = Right
            imgrgb = Left
            if len(imgrgb.shape) >= 3:
                imgrgb = imgrgb[:, :, :3]
                imgrgbR = imgrgbR[:, :, :3]
            else:
                imgrgb = np.expand_dims(imgrgb, -1)
                imgrgbR = np.expand_dims(imgrgbR, -1)
                imgrgb = np.concatenate((imgrgb, imgrgb, imgrgb), -1)
                imgrgbR = np.concatenate((imgrgbR, imgrgbR, imgrgbR), -1)

            imgor = misc.imresize(imgrgb, [h, w, c], 'bicubic')
            imgorR = misc.imresize(imgrgbR, [h, w, c], 'bicubic')

            for fd in range(0, 6):
                if (fd == 0):
                    imggt = Leftgt
                    imggtR = Rightgt
                    img = imgor
                    imgR = imgorR
                elif (fd == 1):
                    imggt = np.rot90(Leftgt, 1)
                    imggtR = np.rot90(Rightgt, 1)
                    img = np.rot90(imgor, 1)
                    imgR = np.rot90(imgorR, 1)
                elif (fd == 2):
                    imggt = np.rot90(Leftgt, 2)
                    imggtR = np.rot90(Rightgt, 2)
                    img = np.rot90(imgor, 2)
                    imgR = np.rot90(imgorR, 2)
                elif (fd == 3):
                    imggt = np.rot90(Leftgt, 3)
                    imggtR = np.rot90(Rightgt, 3)
                    img = np.rot90(imgor, 3)
                    imgR = np.rot90(imgorR, 3)
                elif (fd == 4):
                    imggt = np.flip(Leftgt, 1)
                    imggtR = np.flip(Rightgt, 1)
                    img = np.flip(imgor, 1)
                    imgR = np.flip(imgorR, 1)
                else:
                    imggt = np.flip(Leftgt, 0)
                    imggtR = np.flip(Rightgt, 0)
                    img = np.flip(imgor, 0)
                    imgR = np.flip(imgorR, 0)

                h, w, c = img.shape
                imglgt = imggt
                imgRlgt = imggtR
                imgl = img
                imgRl = imgR

                for i in range(0, h - p + 1, steps):  #
                    for j in range(0, w - p + 1, steps):
                        HRgt = imglgt[i: i + p, j:j + p, :]
                        HRRgt = imgRlgt[i: i + p, j:j + p, :]
                        HR = imgl[i: i + p, j:j + p, :]
                        HRR = imgRl[i: i + p, j:j + p, :]

                        HR = np.float32(HR / 255.0)
                        HRgt = np.float32(HRgt / 255.0)
                        HRR = np.float32(HRR / 255.0)
                        HRRgt = np.float32(HRRgt / 255.0)
                        c1 = np.random.rand()
                        c2 = np.random.rand()
                        if c1 < 0.5:
                            HR, HRR = HR[::-1, :, :], HRR[::-1, :, :]
                            HRgt, HRRgt = HRgt[::-1, :, :], HRRgt[::-1, :, :]
                        if c2 < 0.5:
                            HR, HRR = HR[:, ::-1, :], HRR[:, ::-1, :]
                            HRgt, HRRgt = HRgt[:, ::-1, :], HRRgt[:, ::-1, :]
                        yield HR, HRR, HRgt, HRRgt, sall
    ## include GT images, scoore = 10
    def get_RGBGTallp(self):
        norm = 1
        p = self.patch_size
        steps = p // 2
        mat = sio.loadmat(self.data_score)
        names = mat['name']
        length = len(names[0])
        rank = mat['rank']
        filename = self.filename+self.filenamegt
        random.shuffle(filename)
        for i in range(len(filename)):
            # print(filename[i], self.data_dirgt, filename[i].find(self.data_dirgt))
            if filename[i].find(self.data_dirgt)>=0:
                # print(filename[i])
                namegt = filename[i][len(self.data_dirgt + 'left/'):]
                Left = np.array(misc.imread(self.data_dirgt + 'left/' + namegt))
                Right = np.array(misc.imread(self.data_dirgt + 'right/' + namegt))
                sall = 10 * norm
                name = namegt
                print('GT!!!!!!!!!!!!!   name = ', name, 'namegt = ', namegt)
            else:
                # namemat = names[i].replace(" ", "")  # bics2_Adiron.png  srcnns2_Adiron.png
                name = filename[i][len(self.data_dir + 'left/'):]
                st = name.find('_')
                st1 = name[st + 1:].find('_')
                Left = np.array(misc.imread(self.data_dir + 'left/' + name))
                Right = misc.imread(self.data_dir + 'right/' + name)
                namegt = name[st + st1 + 2:]
                nana = name
                for ni in range(length - len(name)):
                    nana += ' '
                id = list(names).index(nana)
                sall = (rank[0][id]) * norm
                # print('Not GT!!!!!!!!!!!!!   name = ', name, 'namegt = ', namegt)

            # print('name = ', name, 'namegt = ', namegt)
            Leftgt = np.array(misc.imread(self.data_dirgt + 'left/' + namegt))  # filename[i]))
            Rightgt = np.array(misc.imread(self.data_dirgt + 'right/' + namegt))
            h, w, c = Leftgt.shape

            # print('name in filename[i]=', name, ' in mat = ', str(id), 'name in mat names[id]=', names[id], 'rank[id]=', sall)
            imgrgbR = Right
            imgrgb = Left
            if len(imgrgb.shape) >= 3:
                imgrgb = imgrgb[:, :, :3]
                imgrgbR = imgrgbR[:, :, :3]
            else:
                imgrgb = np.expand_dims(imgrgb, -1)
                imgrgbR = np.expand_dims(imgrgbR, -1)
                imgrgb = np.concatenate((imgrgb, imgrgb, imgrgb), -1)
                imgrgbR = np.concatenate((imgrgbR, imgrgbR, imgrgbR), -1)

            imgor = misc.imresize(imgrgb, [h, w, c], 'bicubic')
            imgorR = misc.imresize(imgrgbR, [h, w, c], 'bicubic')

            for fd in range(0, 6):
                if (fd == 0):
                    imggt = Leftgt
                    imggtR = Rightgt
                    img = imgor
                    imgR = imgorR
                elif (fd == 1):
                    imggt = np.rot90(Leftgt, 1)
                    imggtR = np.rot90(Rightgt, 1)
                    img = np.rot90(imgor, 1)
                    imgR = np.rot90(imgorR, 1)
                elif (fd == 2):
                    imggt = np.rot90(Leftgt, 2)
                    imggtR = np.rot90(Rightgt, 2)
                    img = np.rot90(imgor, 2)
                    imgR = np.rot90(imgorR, 2)
                elif (fd == 3):
                    imggt = np.rot90(Leftgt, 3)
                    imggtR = np.rot90(Rightgt, 3)
                    img = np.rot90(imgor, 3)
                    imgR = np.rot90(imgorR, 3)
                elif (fd == 4):
                    imggt = np.flip(Leftgt, 1)
                    imggtR = np.flip(Rightgt, 1)
                    img = np.flip(imgor, 1)
                    imgR = np.flip(imgorR, 1)
                else:
                    imggt = np.flip(Leftgt, 0)
                    imggtR = np.flip(Rightgt, 0)
                    img = np.flip(imgor, 0)
                    imgR = np.flip(imgorR, 0)

                h, w, c = img.shape
                imglgt = imggt
                imgRlgt = imggtR
                imgl = img
                imgRl = imgR

                for i in range(0, h - p + 1, steps):  #
                    for j in range(0, w - p + 1, steps):
                        HRgt = imglgt[i: i + p, j:j + p, :]
                        HRRgt = imgRlgt[i: i + p, j:j + p, :]
                        HR = imgl[i: i + p, j:j + p, :]
                        HRR = imgRl[i: i + p, j:j + p, :]

                        HR = np.float32(HR / 255.0)
                        HRgt = np.float32(HRgt / 255.0)
                        HRR = np.float32(HRR / 255.0)
                        HRRgt = np.float32(HRRgt / 255.0)
                        c1 = np.random.rand()
                        c2 = np.random.rand()
                        if c1 < 0.5:
                            HR, HRR = HR[::-1, :, :], HRR[::-1, :, :]
                            HRgt, HRRgt = HRgt[::-1, :, :], HRRgt[::-1, :, :]
                        if c2 < 0.5:
                            HR, HRR = HR[:, ::-1, :], HRR[:, ::-1, :]
                            HRgt, HRRgt = HRgt[:, ::-1, :], HRRgt[:, ::-1, :]
                        yield HR, HRR, HRgt, HRRgt, sall

    def read_png(self):
        dataset = tf.data.Dataset.from_generator(self.get_RGBGTallp, (tf.float32, tf.float32,tf.float32, tf.float32,tf.float32))
        dataset = dataset.shuffle(self.shuffle_num).prefetch(self.prefetch_num)
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size)).repeat()
        gts, gtsr, im, imr, sc = dataset.make_one_shot_iterator().get_next()
        p = self.patch_size

        imr = tf.reshape(imr, [self.batch_size, p, p, 3])
        im = tf.reshape(im, [self.batch_size, p, p, 3])
        gts = tf.reshape(gts, [self.batch_size, p, p, 3])
        gtsr = tf.reshape(gtsr, [self.batch_size, p, p, 3])
        sc = tf.reshape(sc, [self.batch_size, 1])
        return gts, gtsr, im, imr, sc

############# =================== SISR ====================== ########
class DataLoaderSR():
    def __init__(self, opt):
        self.data_dir = opt.data
        self.filename = get_filenames([self.data_dir + 'left/'])
        random.shuffle(self.filename)
        self.patch_size = opt.patch
        self.scale = opt.scale
        self.batch_size = opt.batch
        self.shuffle_num = 200
        self.prefetch_num = 100
        self.map_parallel_num = 8

    def get_RGBp(self):
        p = self.patch_size
        steps = p // 2
        lp = p // self.scale
        filename = self.filename
        for i in range(len(filename)):
            imgrgb = misc.imread(filename[i])
            name = filename[i][len(self.data_dir + 'left/'):]
            imgrgbR = misc.imread(self.data_dir + 'right/' + name.replace('left', 'right'))

            if len(imgrgb.shape) >= 3:
                imgrgb = imgrgb[:, :, :3]
                imgrgbR = imgrgbR[:, :, :3]
            else:
                imgrgb = np.expand_dims(imgrgb, -1)
                imgrgbR = np.expand_dims(imgrgbR, -1)
                imgrgb = np.concatenate((imgrgb, imgrgb, imgrgb), -1)
                imgrgbR = np.concatenate((imgrgbR, imgrgbR, imgrgbR), -1)
            imgor = imgrgb
            imgorR = imgrgbR

            for fd in range(0, 6):
                if (fd == 0):
                    img = imgor
                    imgR = imgorR
                elif (fd == 1):
                    img = np.rot90(imgor, 1)
                    imgR = np.rot90(imgorR, 1)
                elif (fd == 2):
                    img = np.rot90(imgor, 2)
                    imgR = np.rot90(imgorR, 2)
                elif (fd == 3):
                    img = np.rot90(imgor, 3)
                    imgR = np.rot90(imgorR, 3)
                elif (fd == 4):
                    img = np.flip(imgor, 1)
                    imgR = np.flip(imgorR, 1)
                else:
                    img = np.flip(imgor, 0)
                    imgR = np.flip(imgorR, 0)

                h, w, _ = img.shape
                # print(h, w)
                imgl = misc.imresize(img, 1 / self.scale, 'bicubic')
                imgRl = misc.imresize(imgR, 1 / self.scale, 'bicubic')
                imgb = misc.imresize(imgl, [h,w,3], 'bicubic')
                imgRb = misc.imresize(imgRl, [h,w,3], 'bicubic')

                for i in range(0, h - p + 1, steps):  #
                    for j in range(0, w - p + 1, steps):
                        # print('*****************', img.shape, p, lp, imgol.shape)
                        B = imgb[i: i + p, j:j + p, :]
                        BR = imgRb[i: i + p, j:j + p, :]
                        HR = img[i: i + p, j:j + p, :]
                        HRR = imgR[i: i + p, j:j + p, :]
                        LR = imgl[i // self.scale: i // self.scale + lp, j // self.scale:j // self.scale + lp, :]
                        LRR = imgRl[i // self.scale: i // self.scale + lp, j // self.scale:j // self.scale + lp, :]

                        B = np.float32(B / 255.0)
                        BR = np.float32(BR / 255.0)
                        HR = np.float32(HR / 255.0)
                        HRR = np.float32(HRR / 255.0)
                        LR = np.float32(LR / 255.0)
                        LRR = np.float32(LRR / 255.0)
                        c1 = np.random.rand()
                        c2 = np.random.rand()
                        if c1 < 0.5:
                            HR, LR = HR[::-1, :, :], LR[::-1, :, :]
                            HRR, LRR = HRR[::-1, :, :], LRR[::-1, :, :]
                            BR, B = BR[::-1, :, :], B[::-1, :, :]
                        if c2 < 0.5:
                            HR, LR = HR[:, ::-1, :], LR[:, ::-1, :]
                            HRR, LRR = HRR[:, ::-1, :], LRR[:, ::-1, :]
                            BR, B = BR[:, ::-1, :], B[:, ::-1, :]
                        yield HR, LR, HRR, LRR, B, BR
    def get_RGB_blurp(self):
        p = self.patch_size
        steps = p // 2
        lp = p // self.scale
        filename = self.filename
        for i in range(len(filename)):
            imgrgb = misc.imread(filename[i])
            name = filename[i][len(self.data_dir + 'left/'):]
            imgrgbR = misc.imread(self.data_dir + 'right/' + name.replace('left', 'right'))

            if len(imgrgb.shape) >= 3:
                imgrgb = imgrgb[:, :, :3]
                imgrgbR = imgrgbR[:, :, :3]
            else:
                imgrgb = np.expand_dims(imgrgb, -1)
                imgrgbR = np.expand_dims(imgrgbR, -1)
                imgrgb = np.concatenate((imgrgb, imgrgb, imgrgb), -1)
                imgrgbR = np.concatenate((imgrgbR, imgrgbR, imgrgbR), -1)
            imgor = imgrgb
            imgorR = imgrgbR

            for fd in range(0, 6):
                if (fd == 0):
                    img = imgor
                    imgR = imgorR
                elif (fd == 1):
                    img = np.rot90(imgor, 1)
                    imgR = np.rot90(imgorR, 1)
                elif (fd == 2):
                    img = np.rot90(imgor, 2)
                    imgR = np.rot90(imgorR, 2)
                elif (fd == 3):
                    img = np.rot90(imgor, 3)
                    imgR = np.rot90(imgorR, 3)
                elif (fd == 4):
                    img = np.flip(imgor, 1)
                    imgR = np.flip(imgorR, 1)
                else:
                    img = np.flip(imgor, 0)
                    imgR = np.flip(imgorR, 0)

                h, w, _ = img.shape
                # print(h, w)
                imgl = misc.imresize(img, 1 / self.scale, 'bicubic')
                imgRl = misc.imresize(imgR, 1 / self.scale, 'bicubic')
                imgb = misc.imresize(imgl, [h,w,3], 'bicubic')
                imgRb = misc.imresize(imgRl, [h,w,3], 'bicubic')

                a = random.randint(2, 40) / 10# random.randint(8, 16) / 10  #
                kernel = gaussian_kernel_2d_opencv(a)
                # b = np.float32(kernel)
                imblur = cv2.filter2D(img, -1, kernel)  # signal.convolve(img, kernel)#cv2.filter2D(img, -1, kernel)
                imblurR = cv2.filter2D(imgR, -1, kernel)  # signal.convolve(img, kernel)#cv2.filter2D(img, -1, kernel)
                imglb = misc.imresize(imblur, 1 / self.scale, 'bicubic')
                imglbR = misc.imresize(imblurR, 1 / self.scale, 'bicubic')
                imgbb = misc.imresize(imglb, [h, w, 3], 'bicubic')
                imgbbR = misc.imresize(imglbR, [h, w, 3], 'bicubic')

                for i in range(0, h - p + 1, steps):  #
                    for j in range(0, w - p + 1, steps):
                        # print('*****************', img.shape, p, lp, imgol.shape)
                        B = imgb[i: i + p, j:j + p, :]
                        Bb = np.float32(imgbb[i: i + p, j:j + p, :] / 255.0)
                        BR = imgRb[i: i + p, j:j + p, :]
                        BbR = np.float32(imgbbR[i: i + p, j:j + p, :] / 255.0)
                        HR = img[i: i + p, j:j + p, :]
                        HRR = imgR[i: i + p, j:j + p, :]
                        LR = imgl[i // self.scale: i // self.scale + lp, j // self.scale:j // self.scale + lp, :]
                        LRb = np.float32(imglb[i // self.scale: i // self.scale + lp, j // self.scale:j // self.scale + lp, :] / 255.0)
                        LRR = imgRl[i // self.scale: i // self.scale + lp, j // self.scale:j // self.scale + lp, :]
                        LRbR = np.float32(imglbR[i // self.scale: i // self.scale + lp, j // self.scale:j // self.scale + lp, :] / 255.0)

                        B = np.float32(B / 255.0)
                        BR = np.float32(BR / 255.0)
                        HR = np.float32(HR / 255.0)
                        HRR = np.float32(HRR / 255.0)
                        LR = np.float32(LR / 255.0)
                        LRR = np.float32(LRR / 255.0)
                        c1 = np.random.rand()
                        c2 = np.random.rand()
                        if c1 < 0.5:
                            HR, LR = HR[::-1, :, :], LR[::-1, :, :]
                            HRR, LRR = HRR[::-1, :, :], LRR[::-1, :, :]
                            BR, B = BR[::-1, :, :], B[::-1, :, :]
                            BbR, Bb = BbR[::-1, :, :], Bb[::-1, :, :]
                            LRb, LRbR = LRb[::-1, :, :], LRbR[::-1, :, :]
                        if c2 < 0.5:
                            HR, LR = HR[:, ::-1, :], LR[:, ::-1, :]
                            HRR, LRR = HRR[:, ::-1, :], LRR[:, ::-1, :]
                            BR, B = BR[:, ::-1, :], B[:, ::-1, :]
                            BbR, Bb = BbR[:, ::-1, :], Bb[:, ::-1, :]
                            LRb, LRbR = LRb[:, ::-1, :], LRbR[:, ::-1, :]
                        yield HR, LR, HRR, LRR, B, BR, LRb, LRbR, Bb, BbR
    def get_RGBp_SR_cv18(self):
        p = self.patch_size
        steps = p // 2
        filename = get_filenames([self.data_dir+'left/'])
        for i in range(len(filename)):
            imgrgb = cv2.imread(filename[i]).astype(np.float32) # misc.imread(filename[i])
            name = filename[i][len(self.data_dir + 'left/'):]
            imgrgbR = cv2.imread(self.data_dir + 'right/' + name.replace('left', 'right')).astype(np.float32) # misc.imread(self.data_dir + 'right/' + name.replace('left', 'right'))

            if len(imgrgb.shape) >= 3:
                imgrgbR = imgrgbR[:, :, :3]
                imgrgb = imgrgb[:, :, :3]
            else:
                imgrgb = np.expand_dims(imgrgb, -1)
                imgrgb = np.concatenate((imgrgb, imgrgb, imgrgb), -1)
                imgrgbR = np.expand_dims(imgrgbR, -1)
                imgrgbR = np.concatenate((imgrgbR, imgrgbR, imgrgbR), -1)

            imgycbcr = cv2.cvtColor(imgrgb, cv2.COLOR_BGR2YUV)  # cv2.cvtColor(imgrgb, cv2.COLOR_BGR2YCR_CB)  # rgb2ycbcr(imgrgb)
            imgy = imgycbcr[:, :, :1]
            imgcbcr = imgycbcr[:, :, 1:3]

            imgycbcrR = cv2.cvtColor(imgrgbR, cv2.COLOR_BGR2YUV)  # rgb2ycbcr(imgrgbR)
            imgyR = imgycbcrR[:, :, :1]
            imgcbcrR = imgycbcrR[:, :, 1:3]

            h, w, _ = imgrgb.shape
            imgorc = imgycbcr#imgrgb
            imgorRc = imgycbcrR #imgrgbR
            imgor = imgy
            imgorR = imgyR
            imgc = imgcbcr
            imgcR = imgcbcrR

            for fd in range(0, 6):
                if (fd == 0):
                    img = imgor
                    imgcleft = imgorc
                    imgcr = imgorRc
                    imgR = imgorR
                    cim = imgc
                    cimR = imgcR
                elif (fd == 1):
                    img = np.rot90(imgor, 1)
                    imgcleft = np.rot90(imgorc, 1)
                    imgcr = np.rot90(imgorRc, 1)
                    cim = np.rot90(imgc, 1)
                    cimR = np.rot90(imgcR, 1)
                    imgR = np.rot90(imgorR, 1)
                elif (fd == 2):
                    imgcleft = np.rot90(imgorc, 2)
                    imgcr = np.rot90(imgorRc, 2)
                    img = np.rot90(imgor, 2)
                    cim = np.rot90(imgc, 2)
                    cimR = np.rot90(imgcR, 2)
                    imgR = np.rot90(imgorR, 2)
                elif (fd == 3):
                    imgcleft = np.rot90(imgorc, 3)
                    imgcr = np.rot90(imgorRc, 3)
                    img = np.rot90(imgor, 3)
                    cim = np.rot90(imgc, 3)
                    cimR = np.rot90(imgcR, 3)
                    imgR = np.rot90(imgorR, 3)
                elif (fd == 4):
                    imgcleft = np.flip(imgorc, 1)
                    imgcr = np.flip(imgorRc, 1)
                    img = np.flip(imgor, 1)
                    cim = np.flip(imgc, 1)
                    cimR = np.flip(imgcR, 1)
                    imgR = np.flip(imgorR, 1)
                else:
                    imgcleft = np.flip(imgorc, 0)
                    imgcr = np.flip(imgorRc, 0)
                    img = np.flip(imgor, 0)
                    cim = np.flip(imgc, 0)
                    cimR = np.flip(imgcR, 0)
                    imgR = np.flip(imgorR, 0)

                h, w, _ = img.shape
                imgb = cv2.resize(cv2.resize(img[:, :, 0], (w//self.scale, h//self.scale), interpolation=cv2.INTER_CUBIC), (w, h), interpolation=cv2.INTER_CUBIC)
                imgRb = cv2.resize(cv2.resize(imgR[:, :, 0], (w//self.scale, h//self.scale), interpolation=cv2.INTER_CUBIC), (w, h), interpolation=cv2.INTER_CUBIC)
                imgb = np.expand_dims(imgb, -1)
                imgRb = np.expand_dims(imgRb, -1)

                imgcb = cv2.resize(cv2.resize(cim, (w//self.scale, h//self.scale), interpolation=cv2.INTER_CUBIC), (w, h), interpolation=cv2.INTER_CUBIC)
                imgcbR = cv2.resize(cv2.resize(cimR, (w//self.scale, h//self.scale), interpolation=cv2.INTER_CUBIC), (w, h), interpolation=cv2.INTER_CUBIC)

                right_image = imgRb
                right_input = np.zeros([h, w, 64])
                for k in range(64):
                    right_input[:, :, k:k + 1] = np.roll(right_image, k * 2, 1)
                left_image = imgb
                left_input = np.zeros([h, w, 64])
                for k in range(64):
                    left_input[:, :, k:k + 1] = np.roll(left_image, -k * 2, 1)

                for i in range(0, h - p + 1, steps):  #
                    for j in range(0, w - p + 1, steps):
                        # print('*****************', img.shape, p, lp, imgol.shape)
                        bic = imgb[i: i + p, j:j + p, :]
                        bicr= imgRb[i: i + p, j:j + p, :]
                        bicr64 = right_input[i: i + p, j:j + p, :]
                        bicl64 = left_input[i: i + p, j:j + p, :]
                        bicc = imgcb[i: i + p, j:j + p, :]
                        biccr = imgcbR[i: i + p, j:j + p, :]
                        HR = imgcleft[i: i + p, j:j + p, :]
                        HRR = imgcr[i: i + p, j:j + p, :]

                        bic = np.float32(bic / 255.0)
                        bicr = np.float32(bicr / 255.0)
                        bicr64 = np.float32(bicr64 / 255.0)
                        bicl64 = np.float32(bicl64 / 255.0)
                        bicc = np.float32(bicc / 255.0)
                        biccr = np.float32(biccr / 255.0)
                        HR = np.float32(HR / 255.0)
                        HRR = np.float32(HRR / 255.0)

                        c1 = np.random.rand()
                        c2 = np.random.rand()
                        if c1 < 0.5:
                            HR, HRR = HR[::-1, :, :], HRR[::-1, :, :]
                            # LR, LRR = LR[::-1, :, :], LRR[::-1, :, :]
                            bicc, bicr, bicl64, bicr64, bic, biccr = bicc[::-1, :, :], bicr[::-1, :, :], bicl64[::-1, :, :],\
                                                              bicr64[::-1, :, :], bic[::-1, :, :], biccr[::-1, :, :]
                            # LRc = LRc[::-1, :, :]
                        if c2 < 0.5:
                            HR, HRR = HR[:, ::-1, :], HRR[:, ::-1, :]
                            bicc, bicr, bicl64, bicr64, bic, biccr = bicc[:, ::-1, :], bicr[:, ::-1, :], bicl64[:, ::-1, :],\
                                                              bicr64[:, ::-1, :], bic[:, ::-1, :], biccr[:, ::-1, :]
                        yield HR[:,:,0], HRR[:,:,0], HR, HRR, bic, bicr, bicl64, bicr64, bicc, biccr
                        #gts, gtsr, bc, bcr, bcl64, bcr64, bc_color, bc_color_r

    def read_png(self, srmodel=2, blur=0):
        p = self.patch_size
        lp = p // self.scale
        if srmodel == 2:  ## cvpr18
            dataset = tf.data.Dataset.from_generator(self.get_RGBp_SR_cv18, (
                tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,  tf.float32, tf.float32, tf.float32))
            dataset = dataset.shuffle(self.shuffle_num).prefetch(self.prefetch_num)
            dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size)).repeat()
            gtsy, gtsry, gts, gtsr, bc, bcr, bcl64, bcr64, bc_color, bc_color_r = dataset.make_one_shot_iterator().get_next()
            p = self.patch_size

            bc = tf.reshape(bc, [self.batch_size, p, p, 1])
            bcr = tf.reshape(bcr, [self.batch_size, p, p, 1])
            bcl64 = tf.reshape(bcl64, [self.batch_size, p, p, 64])
            bcr64 = tf.reshape(bcr64, [self.batch_size, p, p, 64])
            bc_color = tf.reshape(bc_color, [self.batch_size, p, p, 2])
            bc_color_r = tf.reshape(bc_color_r, [self.batch_size, p, p, 2])
            gts = tf.reshape(gts, [self.batch_size, p, p, 3])
            gtsr = tf.reshape(gtsr, [self.batch_size, p, p, 3])
            gtsy = tf.reshape(gtsy, [self.batch_size, p, p, 1])
            gtsry = tf.reshape(gtsry, [self.batch_size, p, p, 1])
            return gtsy, gtsry, gts, gtsr, bc, bcr, bcl64, bcr64, bc_color, bc_color_r
        else:
            if blur == 1:
                dataset = tf.data.Dataset.from_generator(self.get_RGB_blurp, (
                    tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                    tf.float32, tf.float32))
                dataset = dataset.shuffle(self.shuffle_num).prefetch(self.prefetch_num)
                dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size)).repeat()
                gts, lrs, gtsr, lrsr, bic, bicr, lrb, lrbr, bicb, bicbr = dataset.make_one_shot_iterator().get_next()

                lrb = tf.reshape(lrb, [self.batch_size, lp, lp, 3])
                lrbr = tf.reshape(lrbr, [self.batch_size, lp, lp, 3])
                lrs = tf.reshape(lrs, [self.batch_size, lp, lp, 3])
                lrsr = tf.reshape(lrsr, [self.batch_size, lp, lp, 3])
                gts = tf.reshape(gts, [self.batch_size, p, p, 3])
                gtsr = tf.reshape(gtsr, [self.batch_size, p, p, 3])
                bic = tf.reshape(bic, [self.batch_size, p, p, 3])
                bicr = tf.reshape(bicr, [self.batch_size, p, p, 3])
                bicb = tf.reshape(bicb, [self.batch_size, p, p, 3])
                bicbr = tf.reshape(bicbr, [self.batch_size, p, p, 3])
                return gts, gtsr, lrs, lrsr, bic, bicr, lrb, lrbr, bicb, bicbr
            else:
                dataset = tf.data.Dataset.from_generator(self.get_RGBp, (
                    tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))
                dataset = dataset.shuffle(self.shuffle_num).prefetch(self.prefetch_num)
                dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size)).repeat()
                gts, lrs, gtsr, lrsr, bic, bicr = dataset.make_one_shot_iterator().get_next()
                p = self.patch_size
                lp = p // self.scale

                lrs = tf.reshape(lrs, [self.batch_size, lp, lp, 3])
                lrsr = tf.reshape(lrsr, [self.batch_size, lp, lp, 3])
                gts = tf.reshape(gts, [self.batch_size, p, p, 3])
                gtsr = tf.reshape(gtsr, [self.batch_size, p, p, 3])
                bic = tf.reshape(bic, [self.batch_size, p, p, 3])
                bicr = tf.reshape(bicr, [self.batch_size, p, p, 3])
                return gts, gtsr, lrs, lrsr, bic, bicr

import scipy.io as sio
def loadmat():
    mat = sio.loadmat('./data/MOS_MidBics32.mat')
    score = mat['score']
    name = mat['name']
    '''
    [[25.761137  13.0618305 20.226454  19.085009  22.37075   21.249632 ]] 
    ['Adirondack_left.bmp' 'Aloe_left.bmp      ' 'Art_left.bmp       '
    'Australia_left.bmp ' 'Baby1_left.bmp     ' 'Baby2_left.bmp     '
    '''
    print(np.mean(score))#, name)

import argparse
def traindata():
    parser = argparse.ArgumentParser(description='method')
    parser.add_argument('--data', default='../../StereoSR/Middlebury/train/fullpng/')
    parser.add_argument('--batch', default=1)
    parser.add_argument('--patch', default=250)
    parser.add_argument('--scale', default=4)
    args = parser.parse_args()
    os.makedirs( './result/trainingdata/', exist_ok=True)

    data_loader = DataLoaderSR(args)
    # new, newLR, newr, newrLR, score = data_loader.read_png()
    # gts, gtsr, bc, bcr, bcl64, bcr64, bc_color, bc_color_r
    new, newr, bic,bic_r, bicr, bicr_l, biccbcr, biccbcr_r = data_loader.read_png(srmodel=2)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        # for itr in range(3):
        #     new1, newLR1, newr1, newrLR1, s = sess.run([new, newLR, newr, newrLR, score])
        #     # # ============================  Lab to RGB  ==================================#
        #     print(s[0])
        #     misc.toimage(new1[0, :, :, :] * 255, high=255, low=0, cmin=0, cmax=255).save(
        #         './result/trainingdata/' + str(itr) + 'newHR.png')
        #     misc.toimage(newr1[0, :, :, :] * 255, high=255, low=0, cmin=0, cmax=255).save(
        #         './result/trainingdata/' + str(itr) + 'newHR-right.png')
        #     misc.toimage(newLR1[0, :, :, :] * 255, high=255, low=0, cmin=0, cmax=255).save(
        #         './result/trainingdata/' + str(itr) + 'newLR.png')
        #     misc.toimage(newrLR1[0, :, :, :] * 255, high=255, low=0, cmin=0, cmax=255).save(
        #         './result/trainingdata/' + str(itr) + 'newLR-right.png')
        for itr in range(1):
            new1, newr1, bc1, bcr, bcr1, bclr, bccbcr1, bccbcrr = sess.run([new, newr, bic,bic_r, bicr, bicr_l, biccbcr, biccbcr_r])
            # # ============================  Lab to RGB  ==================================#
            cv2.imwrite('./result/trainingdata/' + str(itr) + 'GT.png', cv2.cvtColor(new1[0, :, :, :] * 255, cv2.COLOR_YCR_CB2BGR))
            cv2.imwrite('./result/trainingdata/' + str(itr) + 'GT-right.png', cv2.cvtColor(newr1[0, :, :, :] * 255, cv2.COLOR_YCR_CB2BGR))
            bicres = cv2.cvtColor(np.concatenate([bc1[0, :, :, :], bccbcr1[0, :, :, :]],-1) * 255, cv2.COLOR_YCR_CB2BGR)
            bicresr = cv2.cvtColor(np.concatenate([bcr[0, :, :, :], bccbcrr[0, :, :, :]],-1) * 255, cv2.COLOR_YCR_CB2BGR)
            cv2.imwrite('./result/trainingdata/' + str(itr) + 'bic_Ycbcr2RGB.png', bicres)
            cv2.imwrite('./result/trainingdata/' + str(itr) + 'R-bic_Ycbcr2RGB.png', bicresr)
            for k in range(64):
                cv2.imwrite('./result/trainingdata/' + str(itr) + 'bic_Y64_' + str(k) + '-left.png',
                            bclr[0, :, :, k] * 255)

            # misc.toimage(new1[0, :, :, :] * 255, high=255, low=0, cmin=0, cmax=255).save(
            #     './result/trainingdata/' + str(itr) + 'GT.png')
            # misc.toimage(newr1[0, :, :, :] * 255, high=255, low=0, cmin=0, cmax=255).save(
            #     './result/trainingdata/' + str(itr) + 'GT-right.png')
            # bicres = ycbcr2rgb(np.concatenate([bc1[0, :, :, :], bccbcr1[0, :, :, :]],-1) * 255)
            # bicresr = ycbcr2rgb(np.concatenate([bcr[0, :, :, :], bccbcrr[0, :, :, :]],-1) * 255)
            # misc.toimage(bicres, high=255, low=0, cmin=0, cmax=255).save(
            #     './result/trainingdata/' + str(itr) + 'bic_Ycbcr2RGB.png')
            # misc.toimage(bicresr, high=255, low=0, cmin=0, cmax=255).save(
            #     './result/trainingdata/' + str(itr) + 'R_bic_Ycbcr2RGB.png')
            # for k in range(64):
            #     # misc.toimage(bcr1[0, :, :, k] * 255, high=255, low=0, cmin=0, cmax=255).save(
            #     #     './result/trainingdata/' + str(itr) + 'bic_Y64_'+str(k)+'-right.png')
            #     misc.toimage(bclr[0, :, :, k] * 255, high=255, low=0, cmin=0, cmax=255).save(
            #         './result/trainingdata/' + str(itr) + 'bic_Y64_' + str(k) + '-left.png')

if __name__ == '__main__':
    # loadmat()
    traindata()


## ==================================== Keras ================================= #
import scipy
import numpy as np
import random

class DataLoader():
    def __init__(self, dataset_name, path, patch=128):
        self.dataset_name = dataset_name
        # self.img_res = img_res
        self.patch = patch
        self.path = path
        self.rightimgs = get_filenames([self.path+'right/'])
        print('image number = ', len(self.rightimgs))

    def load_dataHR(self, im):  # return all patches in im-th image
        print(im, 'len(batch_images)', len(self.rightimgs))
        img_path = self.rightimgs[im]
        img0 = self.imread(img_path)
        img_left0 = self.imread(img_path.replace('right', 'left'))
        imgs_hr = []
        imgs_hr_left = []
        imgs_lr = []
        imgs_lr_left = []
        for fd in range(6):
            if (fd == 0):
                img = img0
                img_left = img_left0
            elif (fd == 1):
                img = np.rot90(img0, 1)
                img_left = np.rot90(img_left0, 1)
            elif (fd == 2):
                img = np.rot90(img0, 2)
                img_left = np.rot90(img_left0, 2)
            elif (fd == 3):
                img = np.rot90(img0, 3)
                img_left = np.rot90(img_left0, 3)
            elif (fd == 4):
                img = np.flip(img0, 1)
                img_left = np.flip(img_left0, 1)
            elif (fd == 5):
                img = np.flip(img0, 0)
                img_left = np.flip(img_left0, 0)
            h, w, c = img.shape
            for i in range(0, h - self.patch - 1, self.patch // 2):  #
                for j in range(0, w - self.patch - 1, self.patch // 2):
                    img_hr = img[i: i + self.patch, j:j + self.patch, :]
                    img_hr_left = img_left[i: i + self.patch, j:j + self.patch, :]
                    # print(img_hr.shape)
                    img_lr = scipy.misc.imresize(img_hr, 1 / 4, 'bicubic')
                    img_lr_left = scipy.misc.imresize(img_hr_left, 1 / 4, 'bicubic')

                    imgs_hr.append(img_hr)
                    imgs_hr_left.append(img_hr_left)
                    imgs_lr.append(img_lr)
                    imgs_lr_left.append(img_lr_left)
            #         break
            #     break
            # break
        imgs_hr = np.array(imgs_hr) / 255.0
        imgs_hr_left = np.array(imgs_hr_left) / 255.0
        imgs_lr = np.array(imgs_lr) / 255.0
        imgs_lr_left = np.array(imgs_lr_left) / 255.0

        len1 = len(imgs_hr)
        print('patch number in one image', len1)
        return imgs_hr, imgs_lr, imgs_hr_left, imgs_lr_left

    def load_data(self, batch=0, batch_size=1, is_testing=False):
        path = self.path
        rightimgs = get_filenames([path+'right/'])
        batch_images = rightimgs[batch:batch+1]
        print('len(batch_images)', len(batch_images))

        imgs_hr = []
        imgs_hr_left = []
        imgs_lr = []
        imgs_lr_left = []
        for img_path in batch_images:
            img0 = self.imread(img_path)
            img_left0 = self.imread(img_path.replace('right', 'left'))

            fd =random.randint(0, 5)
            if (fd == 0):
                img = img0
                img_left = img_left0
            elif (fd == 1):
                img = np.rot90(img0, 1)
                img_left = np.rot90(img_left0, 1)
            elif (fd == 2):
                img = np.rot90(img0, 2)
                img_left = np.rot90(img_left0, 2)
            elif (fd == 3):
                img = np.rot90(img0, 3)
                img_left = np.rot90(img_left0, 3)
            elif (fd == 4):
                img = np.flip(img0, 1)
                img_left = np.flip(img_left0, 1)
            elif (fd == 5):
                img = np.flip(img0, 0)
                img_left = np.flip(img_left0, 0)
            h, w, c = img.shape

            for i in range(0, h - self.patch - 1, self.patch // 2):  #
                for j in range(0, w - self.patch - 1, self.patch // 2):
                    img_hr = img[i: i + self.patch, j:j + self.patch, :]
                    img_hr_left = img_left[i: i + self.patch, j:j + self.patch, :]
                    # print(img_hr.shape)
                    img_lr = scipy.misc.imresize(img_hr, 1 / 4, 'bicubic')
                    img_lr_left = scipy.misc.imresize(img_hr_left, 1 / 4, 'bicubic')

                    # If training => do random flip
                    if not is_testing and np.random.random() < 0.5:
                        img_hr = np.fliplr(img_hr)
                        img_hr_left = np.fliplr(img_hr_left)
                        img_lr = np.fliplr(img_lr)
                        img_lr_left = np.fliplr(img_lr_left)

                    imgs_hr.append(img_hr)
                    imgs_hr_left.append(img_hr_left)
                    imgs_lr.append(img_lr)
                    imgs_lr_left.append(img_lr_left)

        imgs_hr = np.array(imgs_hr) / 255.0
        imgs_hr_left = np.array(imgs_hr_left) / 255.0
        imgs_lr = np.array(imgs_lr) / 255.0
        imgs_lr_left = np.array(imgs_lr_left) / 255.0

        len1 = len(imgs_hr)
        print('patch number in image', len1)
        bs = random.randint(0, len1-batch_size-1)
        imgs_hr = imgs_hr[bs:bs+batch_size]
        imgs_lr = imgs_lr[bs:bs+batch_size]
        imgs_hr_left = imgs_hr_left[bs:bs+batch_size]
        imgs_lr_left = imgs_lr_left[bs:bs+batch_size]

        return imgs_hr, imgs_lr, imgs_hr_left, imgs_lr_left

    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
