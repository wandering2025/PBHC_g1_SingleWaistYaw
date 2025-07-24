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
    num_mini_batches = 4
    num_learning_epochs = 5
    desired_kl = 0.01
    schedule = "adaptive"
    clip_param = 0.2
    use_clipped_value_loss = True
    l2c2 = {
      'enable': False,
      'lambda_value': 1.0,
      'lambda_policy': 0.1,
    }
    entropy_coef = 0.01
    value_loss_coef = 1.0
    max_grad_norm = 1.0
    num_steps_per_env = 24
    # 下面是storage的配置
    # 这个值是根据奖励函数的数量
    num_rew_fn = 21
    num_act = 23

    
    return {
        'algo_obs_dim_dict': algo_obs_dim_dict,
        'module_dict': module_dict,
        'num_actions': num_actions,
        'init_noise_std': init_noise_std,
        'actor_learning_rate': actor_learning_rate,
        'critic_learning_rate': critic_learning_rate,
        'num_mini_batches': num_mini_batches,
        'num_learning_epochs': num_learning_epochs,
        'desired_kl': desired_kl,
        'schedule': schedule,
        'clip_param': clip_param,
        'use_clipped_value_loss': use_clipped_value_loss,
        'l2c2': l2c2,
        'entropy_coef': entropy_coef,
        'value_loss_coef': value_loss_coef,
        'max_grad_norm': max_grad_norm,
        'num_steps_per_env': num_steps_per_env,
        'num_rew_fn': num_rew_fn,
        'num_act': num_act,
        'encoder': 'GNN'
    }