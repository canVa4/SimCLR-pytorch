from simclr import SimCLR
import yaml
from data_aug.dataset_wrapper import DataSetWrapper
from feature_eval import evaluation
import torch

def main():
    mid = open("config.yaml", "r")
    config = yaml.load(mid, Loader=yaml.FullLoader)
    dataset = DataSetWrapper(config['batch_size'], **config['dataset'])
    simclr = SimCLR(dataset, config)
    simclr.train()
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # folder_name = '../runs/May23_11-24-05_ZX'
    # checkpoints_folder = os.path.join(folder_name, 'checkpoints')
    # evaluation(checkpoints_folder, config, device)


if __name__ == "__main__":
    main()
