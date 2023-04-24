import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from datasets.dataset import MyDataset
import numpy as np
import time
# from lstm_randlabel import LstmNet
from mymodels.mymodel_lstm import MyLSTM
import torch.optim as optim
from tensorboardX import SummaryWriter
import options as opt



if(__name__ == '__main__'):
    opt = __import__('options')
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    logdir = os.path.join(opt.save_folder_root, opt.model_mode, opt.model_name, opt.log_dir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
        # time.sleep(0.001)
    writer = SummaryWriter(log_dir=logdir) # ダッシュボード作成

# データセットをDataLoaderへ入れてDataLoaderの設定をして返す
def dataset2dataloader(dataset, num_workers=opt.num_workers, shuffle=True):
    return DataLoader(dataset,
        batch_size = opt.batch_size,
        shuffle = shuffle,
        num_workers = num_workers, # マルチタスク
        drop_last = True) # Trueにすることでミニバッチから漏れた仲間外れを除去できる (Trueを検討している)

# 学習率を返す
def show_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return np.array(lr).mean()

# テスト
@torch.no_grad()
def test(model, net):
    # テストデータのロード
    dataset = MyDataset(
        opt.data_root
        ,opt.val_list
        ,opt.vid_padding
        ,[opt.rand_x_mean, opt.rand_y_mean]
        ,[opt.rand_x_std, opt.rand_y_std]
        ,opt.input_mode
    )

    #print('num_test_data:{}'.format(len(dataset.data)))
    model.eval() # テストモードへ
    loader = dataset2dataloader(dataset, shuffle=False) # DataLoaderを作成
    loss_list = []
    crit = nn.CrossEntropyLoss()
    tic = time.time()
    for (i_iter, input) in enumerate(loader):
        rand = input.get('rand').cuda()
        label = input.get('label').cuda()
        label = torch.reshape(label, (opt.batch_size,))
        if not opt.is_batch_first:
            rand = rand.view(rand.size(1), rand.size(0), rand.size(2)).contiguous()

        y = net(rand) # ネットへビデオデータを入れる
        # 損出関数での処理
        loss = crit(y, label).detach().cpu().numpy()
        # 損出関数の値を記録
        loss_list.append(loss)
        
        # 結果の文字を入れる
        # 正しい文字列を入れる

        # 条件の回数の時だけエラー率などを表示
        if(i_iter % opt.test_display == 0):
            print(''.join(101*'-'))
            print('{}: {} | {}: {}'.format('predict', torch.argmax(y, dim=1), 'truth', label))
            print(''.join(101*'-'))
            print(''.join(101 *'-'))
            print('test_iter={},loss={:.3f}'.format(i_iter,loss))
            print(''.join(101 *'-'))
    print("\ntest finish.\n")
    print("time: {:.2f} s\n\n".format(time.time() - tic))
    return (np.array(loss_list).mean())

# 訓練
def train(model, net):

    # データのロード
    dataset = MyDataset(
        opt.data_root
        ,opt.train_list
        ,opt.vid_padding
        ,[opt.rand_x_mean, opt.rand_y_mean]
        ,[opt.rand_x_std, opt.rand_y_std]
        ,opt.input_mode
    )

    # DataLoaderの作成
    loader = dataset2dataloader(dataset)
    # optimizerの初期化(Adam使用)
    optimizer = optim.Adam(model.parameters(),
                lr = opt.base_lr,
                weight_decay = opt.decay,#.001,#0.01,#0.1, # パラメータのL2ノルムを正規化としてどれくらい用いるから指定
                eps=1e-8,
                amsgrad = True)# AMSgradを使用する
    """optimizer = optim.SGD(model.parameters(),
                lr = opt.base_lr,
                momentum=0.9,
                )"""

    crit = nn.CrossEntropyLoss()
    tic = time.time()
    # TODO:accuracyの準備
    loss_list = []
    for epoch in range(opt.max_epoch): # epoch分学習する
        epoch_start_tic = time.time()
        for (i_iter, input) in enumerate(loader):
            model.train() # 訓練モードへ
            rand = input.get('rand').cuda()
            label = input.get('label').cuda()
            label = torch.reshape(label, (opt.batch_size,))
            if not opt.is_batch_first:
                rand = rand.view(rand.size(1), rand.size(0), rand.size(2)).contiguous()
            
            optimizer.zero_grad() # パラメータ更新が終わった後の勾配のクリアを行っている。
            y = net(rand) # ビデオデータをネットに投げる
            
            # 損出を求める
            loss = crit(y, label)
            
            loss_list.append(loss)

            # 損出をもとにバックワードで学習
            loss.backward()

            if(opt.is_optimize):
                optimizer.step() # gradプロパティに学習率をかけてパラメータを更新

            tot_iter = i_iter + epoch*len(loader) # 現在のイテレーション数の更新

            # 条件の回数の時、それぞれの経過を表示
            if(tot_iter % opt.train_display == 0):

                writer.add_scalar('train loss', loss, tot_iter)
                print(''.join(101*'-'))
                print('{}: {} | {}: {}'.format('predict', torch.argmax(y, dim=1), 'truth', label))
                print(''.join(101*'-'))
                print(''.join(101*'-'))
                print('epoch={},tot_iter={},base_lr={},loss_mean={:.3f},loss={:.3f}'.format(epoch, tot_iter, opt.base_lr, torch.mean(torch.stack(loss_list)), loss))
                print(''.join(101*'-'))

            if(tot_iter % opt.test_step == 0):
                loss = test(model, net)
                print('i_iter={},lr={},loss={:.3f}'
                    .format(tot_iter,show_lr(optimizer),loss))
                writer.add_scalar('val loss', loss, tot_iter)

            if (tot_iter % opt.save_step == 0):
                savename = 'base_lr{}{}_losst{}_lossv{}.pt'.format(opt.base_lr, opt.save_prefix, torch.mean(torch.stack(loss_list)), loss) # 学習した重みを保存するための名前を作成
                path, name = os.path.split(savename)
                save_path = os.path.join(
                    opt.save_folder_root, opt.model_mode,
                    opt.model_name, path
                )
                name = "losst{}_lossv{}.pt".format(torch.mean(torch.stack(loss_list)), loss)
                if(not os.path.exists(save_path)): 
                    os.makedirs(save_path) # 重みを保存するフォルダを作成する
                    # time.sleep(0.001)
                name = "{}_{}.pt".format(loss, loss)
                save_path = os.path.join(save_path, name)
                
                torch.save(model.state_dict(), save_path) # 学習した重みを保存

            if(not opt.is_optimize):
                exit()
        print("epoch{} finish.".format(epoch))
        print("time {:.2f} s".format(time.time() - epoch_start_tic))
    print("All process finish.")
    print("time: {:.2f} s".format(time.time() - tic))
if(__name__ == '__main__'):
    print("Loading options...")
    model = MyLSTM(
        T=opt.frame_leng
        ,input_size=opt.num_inputs
        ,hidden_size=opt.hidden_size
        ,num_layers=opt.num_layers
        ,bidirectional=opt.bidirectional
        ,dropout_p=opt.dropout_p
        ,is_dropout=opt.is_dropout
    ) # モデルの定義
    model = model.cuda() # gpu使用
    net = nn.DataParallel(model).cuda() # データの並列処理化

    if(hasattr(opt, 'weights')):
        pretrained_dict = torch.load(f=opt.weights)# 学習済みの重みをロード
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
        missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
        print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
        print('miss matched params:{}'.format(missed_params))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    # ネットワークの挙動に再現性を持たせるために、シードを固定して重みの初期値を固定できる
    torch.manual_seed(opt.random_seed)
    torch.cuda.manual_seed_all(opt.random_seed)
    # 訓練開始
    train(model, net)
