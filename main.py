from __future__ import print_function
from tensorboardX import SummaryWriter  # 用于日志记录和可视化
import torch
from config import Config as C  # 全局配置参数
import os
from utils import (
    build_loaders,          # 构建数据加载器
    build_model,            # 构造 EPAN 模型
    train,                  # 训练函数
    evaluate,               # 验证函数（未在 main 中调用）
    get_lr,                 # 获取当前学习率
    save_checkpoint,        # 保存模型检查点
    count_parameters,       # 统计参数量
    set_random_seed         # 设置随机种子
)


def log_train(C, summary_writer, e, loss, lr, teacher_forcing_ratio, scores=None):
    summary_writer.add_scalar(C.tx_train_loss, loss['total'], e)
    summary_writer.add_scalar(C.tx_train_cross_entropy_loss, loss['cross_entropy'], e)
    summary_writer.add_scalar(C.tx_train_contrastive_attention_loss, loss['contrastive_attention'], e)
    summary_writer.add_scalar(C.tx_train_cls_loss, loss['emotion_cls'], e)
    summary_writer.add_scalar(C.tx_lr, lr, e)
    summary_writer.add_scalar(C.tx_teacher_forcing_ratio, teacher_forcing_ratio, e)
    # print("[TRAIN] loss: {} (= CE {} + CA {})".format(
    #     loss['total'], loss['cross_entropy'], loss['contrastive_attention']))
    if scores is not None:
      for metric in C.metrics_full:
          summary_writer.add_scalar("TRAIN SCORE/{}".format(metric), scores[metric], e)
      print("scores: {}".format(scores))


def log_val(C, summary_writer, e, loss, test_vid2GTs, test_vid2pred, vid2idx, scores=None):
    summary_writer.add_scalar(C.tx_val_loss, loss['total'], e)
    summary_writer.add_scalar(C.tx_val_cross_entropy_loss, loss['cross_entropy'], e)
    summary_writer.add_scalar(C.tx_val_contrastive_attention_loss, loss['contrastive_attention'], e)
    summary_writer.add_scalar(C.tx_val_cls_loss, loss['emotion_cls'], e)
    # print("[VAL] loss: {} (= CE {} + CA {})".format(
    #     loss['total'], loss['cross_entropy'], loss['contrastive_attention']))
    if scores is not None:
        for metric in C.metrics_full:
            summary_writer.add_scalar("VAL SCORE/{}".format(metric), scores[metric], e)
        print("scores: {}".format(scores))

def write_logs(C):
    log_path = os.path.join( C.log_dpath, "logs.csv")
    with open(log_path, 'w') as fout:
        fout.write("batchsize,lr,lam,beta,em_wei,filepath,prepath\n")
        line = ', '.join([ str(C.batch_size),str(C.lr),str(C.CA_lambda),str(C.CA_beta),str(C.em_wei),C.file_path,C.pretrained_fpath])
        fout.write("{}\n".format(line))

def get_teacher_forcing_ratio(max_teacher_forcing_ratio, min_teacher_forcing_ratio, epoch, max_epoch):
    x = 1 - float(epoch - 1) / (max_epoch - 1)
    a = max_teacher_forcing_ratio - min_teacher_forcing_ratio
    b = min_teacher_forcing_ratio
    return a * x + b

def get_em_wei(C, vocab): # 加权情感词损失矩阵构建
    weights = torch.ones(vocab.n_vocabs)
    em_wei = weights + C.em_wei
    indexs = []
    em_words = open('EmotionEval/179_words.txt','r')
    for idx,ddd in enumerate(em_words):
        em_word = ddd.split()[0]
        index = vocab.word2idx[em_word]
        indexs.append(index)
    indexs.sort()
    indexs = torch.tensor(indexs)
    weights.scatter_(0, indexs, em_wei) 
    return weights


def main():
    set_random_seed(C.seed)

    summary_writer = SummaryWriter(C.log_dpath)
    # write_logs(C)
    if not os.path.exists(C.result_dir):
        os.makedirs(C.result_dir)

    # 加载数据集（训练、验证、测试、情感训练、情感测试）
    train_iter, val_iter, test_iter, em_train_iter, em_test_iter, vocab = build_loaders(C)

    model = build_model(C, vocab)
    print("#params: ", count_parameters(model))
    model = model.cuda()

    best_val_scores = { 'CFS': -1. }             # 初始化最佳分数
    cross_wei = get_em_wei(C, vocab).cuda()      # 构建情感词损失权重

    print("Pretrained decoder is loading from {}".format(C.pretrained_fpath))    
    path = ""
    # model.load_state_dict(torch.load(C.pretrained_fpath),strict=False) # 加载预训练解码器参数（非严格模式）
    model.load_state_dict(torch.load(path),strict=False) # 加载预训练解码器参数（非严格模式）

    optimizer = torch.optim.Adamax(model.parameters(), lr=C.lr, weight_decay=1e-5)

    for e in range(1, C.epochs + 1):
        print()
        ckpt_fpath = C.ckpt_fpath_tpl_em.format(e)

        """ Train """
        # 动态计算教师强制比率
        teacher_forcing_ratio = get_teacher_forcing_ratio(C.decoder.max_teacher_forcing_ratio,
                                                        C.decoder.min_teacher_forcing_ratio,
                                                        e, C.epochs)
        train_loss = train(e, model, optimizer, em_train_iter, vocab, teacher_forcing_ratio,
                        C, cross_wei)
        # log_train(C, summary_writer, e, train_loss, get_lr(optimizer2), teacher_forcing_ratio)
        log_train(C, summary_writer, e, train_loss, get_lr(optimizer), teacher_forcing_ratio)

        print("Saving checkpoint at epoch={} to {}".format(e, ckpt_fpath))
        # save_checkpoint(ckpt_fpath, e, model, optimizer2)
        save_checkpoint(ckpt_fpath, e, model, optimizer)



if __name__ == "__main__":
    main()

