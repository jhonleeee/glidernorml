import torch
class AgentConfig(object):
    train = True
    debug = True

    scale = 100
    memory_size = 40 * scale
    save_model_step = 10 * scale
    episode_steps = 1 * scale
    explore_steps = 100 * scale
    episode = 1000 * scale
    learn_start = 5 * scale

    max_episode_step = 5000

    model_path = 'model-25.pkl'
    batch_size = 32
    train_frequency = 4
    target_q_update_step = 1 * scale

    layer1_elmts = 20

    history_length = 16

    discount = 0.99

    learning_rate = 0.000025
    ep_min = 0.01
    ep_max = 1.0 if train else ep_min

    double_q = False

    # torch tensor device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    cnn_format = 'NHWC'

class AgentIcoConfig(object):
    train = True
    debug = True

    scale = 100
    memory_size = 40 * scale
    save_model_step = 10 * scale
    episode_steps = 1 * scale
    explore_steps = 100 * scale
    episode = 1000 * scale
    learn_start = 5 * scale

    max_episode_step = 5000

    model_path = 'model-25.pkl'
    batch_size = 32
    train_frequency = 4
    target_q_update_step = 1 * scale

    layer1_elmts = 20

    history_length = 16

    discount = 0.99

    learning_rate = 0.000025
    ep_min = 0.01
    ep_max = 1.0 if train else ep_min

    double_q = False

    # torch tensor device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    cnn_format = 'NHWC'

class EnvironmentConfig(object):
    # step length
    step_len_ms = 10
    state_dim = 4
    action_dim = 5
    debug = True
    action_list = ["+0.0", "-100.0", "+100.0", "+1000.0", "/2.0"]
    init_cwnd = 10
    alpha = 0.2
    beta = 0.6
    delta = 1 - alpha - beta

    # observe and decide
    bad_tput_threshold = 170.0
    last_entries_num = 300
    done_start = 1000
    # currently unused
    train = False
