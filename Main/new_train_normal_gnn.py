import sys
PROJECT_ROOT = sys.argv[1]
sys.path.append(PROJECT_ROOT)

from Utils import io
from Utils.parse_arguments import parse_arguments
from Utils.utils import print_dict, ensure_dir
from Trainer import Trainer
from Wrapper.dataloader import new_build_val_test_dataloader
from Refactor.model import LightGCNWrapper, SAGEWrapper, GATWrapper, GINWrapper, \
    SAGNWrapper, PPRGoWrapper, FAGCNWrapper
from Refactor.dataloader import BlockTrainDataLoader
from Wrapper.lossfunction import BPRLossWrapper, PasserLossWrapper
from Wrapper.optimizer import SGDWrapper, AdamWrapper, DoNothingOptWrapper
from Wrapper.metrics import MetricsWrapper

import setproctitle
import os.path as osp
from collections import defaultdict
import time


def main():
    
    parsed_results = parse_arguments()
    '''
    cmd arg requirements:
    --data_root
    --results_root
    --config_file
    
    '''
    config_file = parsed_results['config_file']  # get's user specified config.
    config = defaultdict(int)
    config.update(io.load_yaml(config_file))
    config.update(parsed_results)
    print_dict(config)
    
    results_root = config['results_root']
    ensure_dir(results_root)
    io.save_yaml(osp.join(results_root, "config.yaml"), dict(config))
    
    if 'str_set' in config:
        config['str_set'] = set(config['str_set'].split())
    
    data = {}
    
    TrainDL = {   # There is only one training data loader. This one will always be used. Not needed for config
        "block_train_dl": BlockTrainDataLoader,
    }[config['train_dl']]
    
    train_dl = TrainDL()
    train_dl.build(data, config)
    val_dl, test_dl = new_build_val_test_dataloader(data, config)
    # Above line simple retrieves val and testing iterators for candidate retrieval or the friend ranking tasks.
    Model = {
        "graphsage": SAGEWrapper,
        "gat": GATWrapper,
        "gin": GINWrapper,
        "sagn": SAGNWrapper,
        "fagcn": FAGCNWrapper,
        "lightgcn": LightGCNWrapper,
        "pprgo": PPRGoWrapper,
    }[config['model']]
    
    model = Model()
    model.build(data, config)
    
    LossFn = {
        "bpr_loss": BPRLossWrapper,
        "do_nothing_loss_passer": PasserLossWrapper,
    }[config['loss_fn']]
    
    loss_fn = LossFn()
    loss_fn.build(data, config)
    
    Opt = {
        "adam": AdamWrapper,
        "sgd": SGDWrapper,
        "do_nothing_opt": DoNothingOptWrapper,
    }[config['optimizer']]
    
    opt = Opt()
    opt.build(model.parameters(), data, config)  # Go back and find what model.parameters are
    
    metrics = MetricsWrapper()
    metrics.build(data, config)
    
    trainer = Trainer(config,
                      model,
                      loss_fn,
                      opt,
                      metrics,
                      train_dl,
                      val_dl,
                      test_dl)
    
    trainer.train()
    
    trainer.test()


if __name__ == "__main__":
    
    setproctitle.setproctitle('xr-gnn-' + 
                              time.strftime("%d%H%M%S", time.localtime(time.time())))
    # In theory I could try to add arguments from here. I would need to pass it in this main
    # I simply copy and past the inputs form the run_lightgcn.sh and run in debug.


    main()


