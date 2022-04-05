import cupy as np

# label 融合
def merge(lab1, n1, lab2, n2):
    buf1 = lab1.ravel()

    # 取出lab1的索引，归一化到从1开始
    idx1 = np.where(buf1)[0]

    # 取lab2与lab1重叠区域的像素
    v1 = buf1[idx1]
    buf2 = lab2.ravel()
    v2 = buf2[idx1]

    # 没有冲突, 直接返回lab2+n1
    if len(idx1)==0 or v2.max()==0:
        return (lab2+n1)*(lab2>0), n1+n2

    v1min = v1.min()-1
    v1 -= v1min

    # 统计lab1， lab2中各个label的面积
    bins1 = np.bincount(v1)
    bins2 = np.bincount(buf2)

    # 构造冲突关系，即都不为0的对应值
    conf = np.array([v1, v2]).T[v2>0]

    # 对冲突关系进行去重，并计算冲突面积
    conf = conf.view(np.int64)
    conf, count = np.unique(conf, return_counts=True)
    conf = conf.view(np.int32).reshape(-1, 2)

    # 计算冲突在各自区域内的占比
    weight1 = bins1[conf[:,0]]
    weight2 = bins2[conf[:,1]]
    k1, k2 = count/weight1, count/weight2

    # 冲突面积达到各自标记面积50%，则视为待解决
    msk = (k1 > 0.5) | (k2>0.5)
    conf, count = conf[msk], count[msk]

    # 按照冲突面积排序，当存在一对多，则解决最严重冲突
    conf = conf[np.argsort(count)][::-1]
    _, idx = np.unique(conf[:,1], True)

    # 构造解决冲突的索引关系
    # print('conf', conf.dtype)
    conf = conf[idx]; conf[:,0]+=v1min
    lut = np.zeros(n2+1, dtype=np.int32)
    lut[conf[:,1]] = conf[:,0]
    lut[lut==0] = np.arange(n1, n1+n2+1-len(conf))
    lut[0] = 0

    # label 融合，把冲突区域清零，切值不相等的地方清零
    lab2 = lut[lab2]
    msk = lab1>0; msk &= lab2>0
    msk &= lab1 != lab2
    lab2 |= lab1; lab2 *= ~msk
    return lab2, n1 + n2 - len(conf)

if __name__ == '__main__':
    from imageio import imread
    import cupyx.scipy.ndimage as ndimg
    import matplotlib.pyplot as plt

    msk1 = np.asarray(imread('lab1.bmp')[:,:,0])
    msk2 = np.asarray(imread('lab2.bmp')[:,:,0])

    lab1, n1 = ndimg.label(msk1, np.ones((3,3)))
    lab1[lab1>0] += 5; n1 += 5
    lab2, n2 = ndimg.label(msk2, np.ones((3,3)))

    labm, n = merge(lab1, n1, lab2, n2)

    plt.subplot(131).imshow(lab1.get())
    plt.subplot(132).imshow(lab2.get())
    plt.subplot(133).imshow(labm.get())

    plt.show()

