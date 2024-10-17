# -*- coding: utf-8 -*-
from utils.mp_utils import ParMap, parallel_monitor, NJOBS

import argparse
import warnings
import os
#from config import *
from archive.load_usr_dataset import load_usr_dataset_by_name
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

import pickle
import torch
from torch.autograd import *
from torch import optim
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader
from utils.base_utils import Queue
from model_utils import *
from dfactor_utils import *
from distance_utils import *

from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale
from utils.base_utils import Debugger, syscmd
__tmat_threshold = 1e-2

def parameterized_gw_torch(x, y, w, torch_dtype, warp=2):
    """
    gw distance in torch with timing factors.
    :param x:
    :param y:
    :param w:
    :param torch_dtype:
    :param warp:
    :return:
    """
    distance = np.sum((x.reshape(x.shape[0], -1, x.shape[1]) - expand_array(y=y, warp=warp)) ** 2,
                      axis=1)
    assert not torch.any(torch.isnan(w)), 'local: {}'.format(w)
    softmin_distance = np.sum(softmax(-distance.astype(np.float64)).astype(np.float32) * distance,
                              axis=1)
    return torch.sqrt(torch.sum(torch.from_numpy(softmin_distance).type(torch_dtype) * torch.abs(w)))


def parameterized_gdtw_torch(x, y, w, torch_dtype, warp=2):
    """
    greedy-dtw distance in torch with timing factors.
    :param x:
    :param y:
    :param w:
    :param torch_dtype:
    :param warp:
    :return:
    """
    dpath = greedy_dtw_path(x=x, y=y, warp=warp)
    return torch.norm((torch.from_numpy(x).type(torch_dtype) * w.reshape(x.shape[0], -1))[dpath[0]] -
                      torch.from_numpy(y[dpath[1]]).type(torch_dtype))


def pattern_distance_torch(pattern, time_series, num_segment, seg_length,
                           local_factor, global_factor, torch_dtype, measurement):
    """
    compute distances between a pattern and a given time series.
    :param pattern:
    :param time_series:
    :param num_segment:
    :param seg_length:
    :param local_factor:
    :param global_factor:
    :param torch_dtype:
    :param measurement:
    :return:
    """
    if measurement == 'gw':
        dist_torch = parameterized_gw_torch
    elif measurement == 'gdtw':
        dist_torch = parameterized_gdtw_torch
    else:
        raise NotImplementedError('unsupported distance {}'.format(measurement))
    assert isinstance(time_series, np.ndarray) and isinstance(pattern, np.ndarray)
    time_series = time_series.reshape(num_segment, seg_length, -1)
    distance = Variable(torch.zeros(num_segment)).type(torch_dtype)
    for k in range(num_segment):
        distance[k] = dist_torch(x=pattern, y=time_series[k], w=local_factor, torch_dtype=torch_dtype)
    return torch.sum(F.softmax(-distance * torch.abs(global_factor), dim=0)
                     * (distance * torch.abs(global_factor)))

def __candidate_cluster_factory(n_clusters, seg_length):
    """
    generate DFactors candidates by clustering.
    :param n_clusters:
    :param seg_length:
    :return:
    """
    def __main__(pid, args, queue):
        ret = []
        for time_series_segments in args:
            kmeans = KMeans(n_clusters=n_clusters).fit(time_series_segments)
            ret.append(kmeans.cluster_centers_.reshape(n_clusters, seg_length, -1))
            queue.put(0)
        return ret
    return __main__
def __candidate_greedy_factory(n_candiates, seg_length):
    """
    generate DFactors candidates by greedy algorithms.
    :param n_candiates:
    :param seg_length:
    :return:
    """
    def __main__(pid, args, queue):
        ret = []
        for time_series_segments in args:
            size = time_series_segments.shape[0]
            center_segment = np.mean(time_series_segments, axis=0)
            cand_dist = np.linalg.norm(
                time_series_segments.reshape(size, -1) - center_segment.reshape(1, -1), axis=1)
            tmp = []
            for cnt in range(n_candiates):
                idx = np.argmax(cand_dist)
                cand_dist[idx] = -1
                update_idx = cand_dist >= 0
                dims = np.sum(update_idx)
                cand_dist[update_idx] += np.linalg.norm(
                    time_series_segments[update_idx].reshape(dims, -1) - time_series_segments[idx].reshape(1, -1),
                    axis=1
                )
                tmp.append(time_series_segments[idx].reshape(seg_length, -1))
            ret.append(tmp)
            queue.put(0)
        return ret
    return __main__
def __DFactor_candidate_loss(cand, time_series_set, label, num_segment, seg_length,
                            data_size, p, lr, alpha, beta, num_batch, gpu_enable,
                            measurement, **kwargs):
    """
    loss for learning time-aware DFactors.
    :param cand:
    :param time_series_set:
    :param label:
    :param num_segment:
    :param seg_length:
    :param data_size:
    :param p:
        normalizing parameter (0, 1, or 2).
    :param lr:
        learning rate.
    :param alpha:
        penalty weight for local timing factor.
    :param beta:
        penalty weight for global timing factor.
    :param num_batch:
    :param gpu_enable:
    :param measurement:
    :param kwargs:
    :return:
    """
    if gpu_enable:
        torch_dtype = torch.cuda.FloatTensor
    else:
        torch_dtype = torch.FloatTensor
    dataset_numpy = NumpyDataset(time_series_set, label)
    num_class = len(np.unique(label).reshape(-1))
    batch_size = int(len(dataset_numpy) // num_batch)
    local_factor_variable = Variable(torch.ones(seg_length).type(torch_dtype) / seg_length, requires_grad=True)
    global_factor_variable = Variable(torch.ones(num_segment).type(torch_dtype) / num_segment, requires_grad=True)
    current_loss, loss_queue, cnt, nan_cnt = 0.0, Queue(max_size=int(num_batch * 0.2)), 0, 0
    current_main_loss, current_penalty_loss = 0.0, 0.0
    max_iters, optimizer = kwargs.get('max_iters', 1), kwargs.get('optimizer', 'Adam')
    if optimizer == 'Adam':
        optimizer = optim.Adam
    elif optimizer == 'Adadelta':
        optimizer = optim.Adadelta
    elif optimizer == 'Adamax':
        optimizer = optim.Adamax
    else:
        raise NotImplementedError('unsupported optimizer {} for time-aware DFactors learning'.format(optimizer))
    optimizer = optimizer([local_factor_variable, global_factor_variable], lr=lr)

    while cnt < max_iters:
        sampler = StratifiedSampler(label=label, num_class=num_class)
        dataloader = DataLoader(dataset=dataset_numpy, batch_size=batch_size, sampler=sampler)
        batch_cnt = 0
        for x, y in dataloader:
            x = np.array(x, dtype=np.float32).reshape(len(x), -1, data_size)
            y = np.array(y, dtype=np.float32).reshape(-1)
            assert not np.any(np.isnan(x)), 'original time series data with nan'
            lb_idx, sample_flag = [], True
            for k in range(num_class):
                tmp_idx = np.argwhere(y == k).reshape(-1)
                if k >= 1 and len(tmp_idx) > 0:
                    sample_flag = False
                lb_idx.append(tmp_idx)
            if len(lb_idx[0]) == 0 or sample_flag:
                Debugger.debug_print('weighted sampling exception, positive {:.2f}/{}'.format(np.sum(y)/len(y), len(y)))
                continue
            loss = torch.Tensor([0.0]).type(torch_dtype)
            main_loss = torch.Tensor([0.0]).type(torch_dtype)
            penalty_loss = torch.Tensor([0.0]).type(torch_dtype)
            dist_tensor = torch.zeros(x.shape[0]).type(torch_dtype)
            for k in range(x.shape[0]):
                dist_tensor[k] = pattern_distance_torch(
                    pattern=cand, time_series=x[k, :, :], num_segment=num_segment,
                    seg_length=seg_length, local_factor=local_factor_variable,
                    global_factor=global_factor_variable, torch_dtype=torch_dtype,
                    measurement=measurement
                    # ignore the warning of reshape/view for local_factor_variable
                )
            assert not torch.isnan(dist_tensor).any(), 'dist: {}\nlocal: {}\nglobal: {}'.format(
                dist_tensor, local_factor_variable, global_factor_variable)
            mean, std = torch.mean(dist_tensor), torch.std(dist_tensor)
            dist_tensor = (dist_tensor - mean) / std
            # Debugger.info_print('transform: {}, {}'.format(torch.max(dist_tensor), torch.min(dist_tensor)))
            # Debugger.time_print(msg='pattern distance', begin=begin, profiling=True)
            for k in range(1, len(lb_idx)):
                src = dist_tensor[lb_idx[0]]
                dst = dist_tensor[lb_idx[k]]
                loss -= torch.abs(torch.distributions.kl.kl_divergence(
                    Normal(torch.mean(src), torch.std(src)),
                    Normal(torch.mean(dst), torch.std(dst))))
                main_loss -= torch.abs(torch.distributions.kl.kl_divergence(
                    Normal(torch.mean(src), torch.std(src)),
                    Normal(torch.mean(dst), torch.std(dst))))
                # Debugger.info_print('KL-loss: {}'.format(loss))
            loss += (alpha * torch.norm(local_factor_variable, p=p) / seg_length)
            loss += (beta * torch.norm(global_factor_variable, p=p) / num_segment)

            penalty_loss += (alpha * torch.norm(local_factor_variable, p=p) / seg_length)
            penalty_loss += (beta * torch.norm(global_factor_variable, p=p) / num_segment)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if gpu_enable:
                current_loss = float(loss.cpu().data.numpy())
                current_main_loss = float(main_loss.cpu().data)
                current_penalty_loss = float(penalty_loss.cpu().data)
            else:
                current_loss = float(loss.data.numpy())
                current_main_loss = float(main_loss.data)
                current_penalty_loss = float(penalty_loss.data)
            loss_queue.enqueue(current_loss)
            if np.isnan(current_loss) or torch.any(torch.isnan(local_factor_variable))\
                    or torch.any(torch.isnan(global_factor_variable)):
                local_factor_variable = Variable(torch.ones(seg_length).type(torch_dtype) / seg_length, requires_grad=True)
                global_factor_variable = Variable(torch.ones(num_segment).type(torch_dtype) / num_segment, requires_grad=True)
                current_loss = 1e5
                nan_cnt += 1
                if nan_cnt >= max_iters:
                    break
            else:
                Debugger.debug_print('{:.2f}% steps, loss {:.6f} with {:.6f} and penalty {:.6f}'.format(
                    batch_cnt * 100 / num_batch, current_loss, current_main_loss, current_penalty_loss))
            batch_cnt += 1
        cnt += 1
        if nan_cnt >= max_iters:
            break
        else:
            avg_loss = np.mean(loss_queue.queue[1:])
            if abs(current_loss - avg_loss) < kwargs.get('epsilon', 1e-2):
                break
    local_factor_variable = torch.abs(local_factor_variable)
    global_factor_variable = torch.abs(global_factor_variable)
    if gpu_enable:
        local_factor = local_factor_variable.cpu().data.numpy()
        global_factor = global_factor_variable.cpu().data.numpy()
    else:
        local_factor = local_factor_variable.data.numpy()
        global_factor = global_factor_variable.data.numpy()
    return local_factor, global_factor, current_loss, current_main_loss, current_penalty_loss




def __DFactor_candidate_loss_factory(time_series_set, label, num_segment,
                                    seg_length, data_size, p, lr, alpha, beta, num_batch,
                                    gpu_enable, measurement, **kwargs):
    """
    paralleling compute DFactor losses.
    :param time_series_set:
    :param label:
    :param num_segment:
    :param seg_length:
    :param data_size:
    :param p:
    :param lr:
    :param alpha:
    :param beta:
    :param num_batch:
    :param gpu_enable:
    :param measurement:
    :param kwargs:
    :return:
    """
    def __main__(pid, args, queue):
        ret = []
        for cand in args:
            local_factor, global_factor, loss, main_loss, penalty = __DFactor_candidate_loss(
                cand=cand, time_series_set=time_series_set, label=label, num_segment=num_segment,
                seg_length=seg_length, data_size=data_size, p=p, lr=lr,
                alpha=alpha, beta=beta, num_batch=num_batch, gpu_enable=gpu_enable,
                measurement=measurement, **kwargs
            )
            ret.append((cand, local_factor, global_factor, loss, main_loss, penalty))
            queue.put(0)
        return ret
    return __main__

def generate_DFactor_candidate(time_series_set, num_segment, seg_length, candidate_size, **kwargs):
    """
    generate DFactor candidates.
    :param time_series_set:
    :param num_segment:
    :param seg_length:
    :param candidate_size:
    :param kwargs:
        candidate_method: 'greedy' or 'cluster'.
        debug: bool.
    :return:
    """
    __method, __debug = kwargs.get('candidate_method', 'greedy'), kwargs.get('debug', True)
    njobs = kwargs.get('njobs', NJOBS)
    Debugger.debug_print('begin to generate DFactor candidates...', __debug)
    num_time_series = time_series_set.shape[0]
    time_series_set = time_series_set.reshape(num_time_series, num_segment, seg_length, -1)
    assert candidate_size >= num_segment, 'candidate-size {} should be larger ' \
                                            'than n_segments {}'.format(candidate_size, num_segment)
    args, n_clusters = [], candidate_size // num_segment
    for idx in range(num_segment):
        args.append(time_series_set[:, idx, :, :].reshape(num_time_series, -1))
    if __method == 'cluster':
        work_func = __candidate_cluster_factory
    elif __method == 'greedy':
        work_func = __candidate_greedy_factory
    else:
        raise NotImplementedError('unsupported candidate generating method {}'.format(__method))
    parmap = ParMap(
        work=work_func(n_clusters, seg_length),
        monitor=parallel_monitor(msg='generate candidate by {}'.format(__method),
                                    size=num_segment, debug=__debug),
        njobs=njobs
    )
    ret = np.concatenate(parmap.run(data=args), axis=0)
    Debugger.info_print('candidates with length {} sampling done...'.format(seg_length))
    Debugger.info_print('totally {} candidates with shape {}'.format(len(ret), ret.shape))
    return ret

def learn_time_aware_DFactors(time_series_set, label, K, C, num_segment, seg_length, data_size,
                            p, lr, alpha, beta, num_batch, gpu_enable, measurement, **kwargs):
    """
    learn time-aware DFactors.
    :param time_series_set:
        input time series data.
    :param label:
        input label.
    :param K:
        number of DFactors that finally learned.
    :param C:
        number of DFactor candidates in learning procedure.
    :param num_segment:
    :param seg_length:
    :param data_size:
    :param p:
    :param lr:
    :param alpha:
    :param beta:
    :param num_batch:
    :param gpu_enable:
    :param measurement:
    :param kwargs:
    :return:
    """
    cands = generate_DFactor_candidate(time_series_set=time_series_set, num_segment=num_segment,
                                        seg_length=seg_length, candidate_size=C, **kwargs)
    parmap = ParMap(
        work=__DFactor_candidate_loss_factory(
            time_series_set=time_series_set, label=label, num_segment=num_segment, seg_length=seg_length,
            data_size=data_size, p=p, lr=lr, alpha=alpha, beta=beta, num_batch=num_batch,
            gpu_enable=gpu_enable, measurement=measurement, **kwargs
        ),
        monitor=parallel_monitor(msg='learning time-aware DFactors', size=len(cands),
                                    debug=kwargs.get('debug', True)),
        njobs=kwargs.get('njobs', NJOBS)
    )
    result = sorted(parmap.run(data=cands), key=lambda x: x[3])
    ret = []
    for (cand, local_factor, global_factor, loss, main_loss, penalty) in result:
        ret.append((cand, local_factor, global_factor, loss))
    return sorted(ret, key=lambda x: x[-1])[:K]


def learn_DFactors(x, y, num_segment, data_size, num_batch,
                   kernel, K, C, seg_length,
                   opt_metric, init, gpu_enable,
                   warp, tflag, mode,
                   percentile, candidate_method,
                   batch_size, njobs,
                   optimizer, alpha,
                   beta, measurement,
                   representation_size, 
                   scaled, norm, global_flag, n_splits):
    """
    learn time-aware DFactors.
    :param x:
        input time series data.
    :param y:
        input label.
    :param num_segment:
        number of segments that time series are divided into.
    :param data_size:
        data dimension of time series.
    :param num_batch:
        number of batch in training.
    :return:
    """
    p=2
    lr=1e-2
    assert x.shape[1] == num_segment * seg_length
    if tflag:
        DFactors = learn_time_aware_DFactors(
            time_series_set=x, label=y, K=K, C=C, p=p,
            num_segment=num_segment, seg_length=seg_length, data_size=data_size,
            lr=lr, alpha=alpha, beta=beta, num_batch=num_batch,
            measurement=measurement, gpu_enable=gpu_enable)
    else:
            raise NotImplementedError()
    return DFactors
def save_DFactors(self, fpath):
    pickle.dump(self.DFactors, open(fpath, 'wb'))


if __name__ == '__main__':
    warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='ucr-Deepfacegen',help='ucr-Earthquakes/WormsTwoClass/Strawberry')
    parser.add_argument('--K', type=int, default=100, help='number of DFactors extracted')
    parser.add_argument('--C', type=int, default=800, help='number of DFactor candidates')
    parser.add_argument('--n_splits', type=int, default=5, help='number of splits in cross-validation')
    parser.add_argument('--num_segment', type=int, default=12, help='number of segment a time series is divided into')
    parser.add_argument('--seg_length', type=int, default=30, help='segment length')
    parser.add_argument('--njobs', type=int, default=8, help='number of threads in parallel')
    parser.add_argument('--data_size', type=int, default=1, help='data dimension of time series')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer used in time-aware DFactors learning')
    parser.add_argument('--alpha', type=float, default=0.1, help='penalty parameter of local timing factor')
    parser.add_argument('--beta', type=float, default=0.05, help='penalty parameter of global timing factor')
    parser.add_argument('--init', type=int, default=0, help='init index of time series data')
    parser.add_argument('--gpu_enable', action='store_true', default=False, help='bool, whether to use GPU')
    parser.add_argument('--opt_metric', type=str, default='accuracy', help='which metric to optimize in prediction')
    #parser.add_argument('--cache', action='store_true', default=False, help='whether to dump model to local file')
    parser.add_argument('--embed', type=str, default='aggregate',help='which embed strategy to use (aggregate/concatenate)')
    parser.add_argument('--embed_size', type=int, default=256, help='embedding size of DFactors')
    parser.add_argument('--warp', type=int, default=2, help='warping size in greedy-dtw')
    parser.add_argument('--cmethod', type=str, default='greedy',help='which algorithm to use in candidate generation (cluster/greedy)')
    parser.add_argument('--kernel', type=str, default='xgb', help='specify outer-classifier (default xgboost)')
    parser.add_argument('--percentile', type=int, default=10,help='percentile for distance threshold in constructing graph')
    parser.add_argument('--measurement', type=str, default='gdtw',help='which distance metric to use (default greedy-dtw)')
    parser.add_argument('--batch_size', type=int, default=500,help='batch size in each training step')
    parser.add_argument('--tflag', action='store_false', default=True, help='flag that whether to use timing factors')
    parser.add_argument('--scaled', action='store_true', default=False, help='flag that whether to rescale time series data')
    parser.add_argument('--norm', action='store_true', default=False,help='flag that whether to normalize extracted representations')
    parser.add_argument('--no_global', action='store_false', default=True,help='whether to use global timing factors')
    parser.add_argument('--quantitative', type=str, default='Seq_SSIM',help='quantitative value')
    parser.add_argument('--seg', type=str, default='seg0',help='segment value')
    parser.add_argument('--output_path', type=str, default='/data/usr/lhr/Time_DFactor/Shaplet_global/DFactor_cache',help='保存路径')
    args = parser.parse_args()
    Debugger.info_print('running with {}'.format(args.__dict__))

    if args.dataset.startswith('ucr'):
        dataset = args.dataset.rstrip('\n\r').split('-')[-1]
        quantitative=args.quantitative
        #print(dataset)
        x_train, y_train, x_test, y_test = load_usr_dataset_by_name(
            fname=dataset, length=args.seg_length * args.num_segment,quantitative=quantitative,seg=args.seg)
    else:
        raise NotImplementedError()
    #正样本比例
    Debugger.info_print('training: {:.2f} positive ratio with {}'.format(float(sum(y_train) / len(y_train)),
                                                                         len(y_train)))
    Debugger.info_print('test: {:.2f} positive ratio with {}'.format(float(sum(y_test) / len(y_test)),
                                                                     len(y_test)))

    learned_DFactors=learn_DFactors(
                        x=x_train, y=y_train, num_segment=int(x_train.shape[1] / args.seg_length),
                        data_size=args.data_size, num_batch=int(x_train.shape[0] // args.batch_size),
                        kernel=args.kernel, K=args.K, C=args.C, seg_length=args.seg_length,
                        opt_metric=args.opt_metric, init=args.init, gpu_enable=args.gpu_enable,
                        warp=args.warp, tflag=args.tflag, mode=args.embed,
                        percentile=args.percentile, candidate_method=args.cmethod,
                        batch_size=args.batch_size, njobs=args.njobs,
                        optimizer=args.optimizer, alpha=args.alpha,
                        beta=args.beta, measurement=args.measurement,
                        representation_size=args.embed_size, 
                        scaled=args.scaled, norm=args.norm, global_flag=args.no_global,n_splits=args.n_splits)
    # 构造保存文件的完整路径
    dir_path = args.output_path
    fpath = os.path.join(dir_path, f'{args.seg}.cache')
    
    # 如果目录不存在，创建目录
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    
    # 序列化并保存 learned_DFactors
    with open(fpath, 'wb') as f:
        pickle.dump(learned_DFactors, f)

    print(f"DFactor 已保存到 {fpath}")