def get_config():
    algo_obs_dim_dict = {
        'actor_obs': 380,
        'critic_obs': 630,
    }
    
    module_dict = {
        'actor': {
            'input_dim': ['actor_obs'],
            'output_dim': ['robot_action_dim'],
            'layer_config': {
                'type': 'MLP',
                'hidden_dims': [512, 256, 128],
                'activation': 'ELU',
            }
        },
        'critic': {
            'input_dim': ['critic_obs'],
            'output_dim': [21],
            'layer_config': {
                'type': 'MLP',
                'hidden_dims': [768, 512, 128],
                'activation': 'ELU',
            }
        }
    }
    num_actions = 23
    init_noise_std = 0.8
    actor_learning_rate = 1.e-3
    critic_learning_rate = 1.e-3
    return {
        'algo_obs_dim_dict': algo_obs_dim_dict,
        'module_dict': module_dict,
        'num_actions': num_actions,
        'init_noise_std': init_noise_std,
        'actor_learning_rate': actor_learning_rate,
        'critic_learning_rate': critic_learning_rate
    }