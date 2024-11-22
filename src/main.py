import argparse
import json
from model import *
from utils import *
from train import *
from verify import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="config file location", type=str)
    args = parser.parse_args()
    with open(args.config_file, 'r') as f:
        config = json.load(f)

    if config['model_type'] == 'conv':
        model = ConvVAE
    elif config['model_type'] == 'fc':
        model = FCVAE
    elif config['model_type'] == 'fire':
        model = FIREVAE
    else:
        raise ValueError('model_type not recognized')
    model = model(config['latent_dim'], config['input_dim'], config['num_channels'], config['kernel_num'], ibp = True)
    train_model(model, 
                model_name=config['model_name'],
                dataset=config['dataset'], 
                epochs=config.get('epochs', 100),
                batch_size=config.get('batch_size', 32), 
                lr=config.get('lr', 1e-4),
                weight_decay=config.get('weight_decay', 1e-5),
                checkpoint_dir=config.get('checkpoint_dir', './models'),
                device=config.get('device', 'cuda'),
                training_type=config.get('training_type', 'standard'),
                epsilon=config.get('epsilon', 0.1),
                pgd_step_size=config.get('step_size', 0.01),
                pgd_iterations=config.get('iterations', 5),
                deltas=config.get('deltas', [0.05]),
                civet_standard_iters=config.get('civet_standard_iters', 250),
                civet_rampup_iters=config.get('civet_rampup_iters', 250),
                max_bs_depth=config.get('max_bs_depth', 20),
                loss_scaling=config.get('loss_scaling', 1),
                tau=config.get('tau', 0.1))
    
    test_dataset = get_dataloader(config['dataset'])['test']
    if config['model_type'] == 'fire':
        baseline_perf, rafa, ibplatent = verify_fire(model,
                                                    test_dataset, 
                                                    config.get('device', 'cuda'), 
                                                    config.get('eval_step_size', 0.001), 
                                                    config.get('epsilon', 0.1), 
                                                    config.get('eval_iterations', 40))
        print(f'Baseline Performance: {baseline_perf}')
        print(f'RAFA Performance: {rafa}')
        torch.save(ibplatent, config['ibplatent_path'])
    else:
        baseline_perf, mda, lsa, ibplatent = verify(model,
                                                    test_dataset, 
                                                    config.get('device', 'cuda'), 
                                                    config.get('eval_step_size', 0.001), 
                                                    config.get('epsilon', 0.1), 
                                                    config.get('eval_iterations', 40))
        print(f'Baseline Performance: {baseline_perf}')
        print(f'MDA Performance: {mda}')
        print(f'LSA Performance: {lsa}')
        torch.save(ibplatent, config['ibplatent_path'])
