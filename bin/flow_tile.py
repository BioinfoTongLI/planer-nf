#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import cupy as np, cupyx.scipy.ndimage as ndimg
import math, itertools
from label_merge import merge
import fire

def make_slice(l, w, mar):
    r = np.linspace(0, l-w, math.ceil((l-mar)/(w-mar)))
    return [slice(i, i+w) for i in r.astype(int).tolist()]

def grid_slice(H, W, h, w, mar):
    a, b = make_slice(H, h, mar), make_slice(W, w, mar)
    return list(itertools.product(a, b))


def estimate_volumes(arr, sigma=3):
    msk = arr > 50;
    idx = np.arange(len(arr), dtype=np.uint32)
    idx, arr = idx[msk], arr[msk]
    for k in np.linspace(5, sigma, 5):
       std = arr.std()
       dif = np.abs(arr - arr.mean())
       msk = dif < std * k
       idx, arr = idx[msk], arr[msk]
    return arr.mean(), arr.std()

def flow2msk(flowp, level=0.5, grad=0.5, area=None, volume=None):
    shp, dim = flowp.shape[:-1], flowp.ndim - 1
    l = np.linalg.norm(flowp[:,:,:2], axis=-1)
    flow = flowp[:,:,:2]/l.reshape(shp+(1,))
    flow[(flowp[:,:,2]<level)|(l<grad)] = 0
    ss = ((slice(None),) * (dim) + ([0,-1],)) * 2
    for i in range(dim): flow[ss[dim-i:-i-2]]=0
    sn = np.sign(flow); sn *= 0.5; flow += sn;
    dn = flow.astype(np.int32).reshape(-1, dim)
    strides = np.cumprod(np.array((1,)+shp[::-1]))
    dn = (strides[-2::-1] * dn).sum(axis=-1)
    rst = np.arange(flow.size//dim, dtype='uint32')
    np.add(rst, dn, out=rst, casting='unsafe')
    for i in range(10): rst = rst[rst]
    hist = np.bincount(rst, None, len(rst))
    hist = hist.astype(np.uint32).reshape(shp)
    lab, n = ndimg.label(hist, np.ones((3,)*dim))
    volumes = ndimg.sum(hist, lab, np.arange(n+1))
    areas = np.bincount(lab.ravel())
    mean, std = estimate_volumes(volumes, 2)
    if not volume: volume = max(mean-std*3, 50)
    if not area: area = volumes // 3
    msk = (areas<area) & (volumes>volume)
    lut = np.zeros(n+1, np.uint32)
    lut[msk] = np.arange(1, msk.sum()+1)
    return lut[lab].ravel()[rst].reshape(shp)

# 去除边界像素，会干扰融合
def remove_edge(lab):
    ss = ((slice(None),) * (lab.ndim) + ([1,-2],)) * 2
    cs, dim = [], lab.ndim
    cs = [lab[ss[dim-i:-i-2]] for i in range(dim)]
    cs = np.concatenate([i.ravel() for i in cs])
    cs = np.unique(cs)[1:].astype('int32')
    lut = np.ones(int(lab.max())+1, dtype=np.int32)
    lut[cs] = 0
    lut[lut>0] = np.arange(len(lut)-len(cs))
    return lut[lab]


def main(stem, flow_npy):
    from tifffile import imread, imwrite
    from time import time

    # flow = imread('flow.tif').astype('float16')

    flow = np.load(flow_npy, allow_pickle=True).astype('float16')
    # flow = np.tile(flow, (6,10,1))

    paper = np.zeros(flow.shape[:2], dtype=np.int32)
    ss = grid_slice(*flow.shape[:2], 4096, 4096, 100)


    rst, n = [], 0
    start = time()
    for sli in ss:
        print(ss.index(sli), len(ss))
        cpflow = np.asarray(flow[sli])
        cppaper = paper[sli]

        lab = flow2msk(cpflow)
        lab = remove_edge(lab)
        n1, n2 = int(n), int(lab.max())
        lab, n = merge(cppaper, n1, lab, n2)
        paper[sli] = lab
    print('total cost:', time()-start)

    del flow; msk = paper > 0
    # paper %= 8; paper += 1; paper *= msk
    paper = paper.get()

    imwrite(f'{stem}_lab.tif', paper.astype(np.uint32))

if __name__ == "__main__":
    fire.Fire(main)
