from rltesting.utils.logger import Logger
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from rltesting.torch_rl.dreamer.dreamer_atari import *
from rltesting.torch_rl.utils import load_config, simple_process_config

def load_encoder(dreamer, encoder_path):
    pretrained_encoder_state = torch.load(encoder_path, map_location="cpu", weights_only=False)
    wm_encoder_state = dreamer.world_model.encoder.state_dict()
    enc_keys = list(pretrained_encoder_state.keys())
    wm_enc_keys = list(wm_encoder_state.keys())
    for i in range(len(enc_keys)):
        if wm_encoder_state[wm_enc_keys[i]].shape == pretrained_encoder_state[enc_keys[i]].shape:
            wm_encoder_state[wm_enc_keys[i]] = pretrained_encoder_state[enc_keys[i]]
    dreamer.world_model.encoder.load_state_dict(wm_encoder_state)

def load_decoder(dreamer, decoder_path):
    pretrained_decoder_state = torch.load(decoder_path, map_location="cpu", weights_only=False)
    wm_decoder_state = dreamer.world_model.decoder.state_dict()
    dec_keys = list(pretrained_decoder_state.keys())
    wm_dec_keys = list(wm_decoder_state.keys())
    for i in range(len(dec_keys)):
        if wm_decoder_state[wm_dec_keys[i]].shape == pretrained_decoder_state[dec_keys[i]].shape:
            wm_decoder_state[wm_dec_keys[i]] = pretrained_decoder_state[dec_keys[i]]
    dreamer.world_model.decoder.load_state_dict(wm_decoder_state)
        
def process_config(config):
    config = simple_process_config(config)
    if config.act == "silu":
        config.act = nn.SiLU
    elif config.act == "gelu":
        config.act = nn.GELU
    elif config.act == "relu":
        config.act = nn.ReLU
    if config.obs_type == "image":
        config.image_dim = (config.num_channels, config.image_size, config.image_size)
        size = config.image_size // 2 ** (config.num_convs)
        config.output_dim = (config.filter_base * 2 ** (config.num_convs - 1), size, size)
    config.latent_size = config.num_categoricals * config.num_codes
    config.state_size = config.hidden_state_size + config.latent_size
    config.train_every = config.sample_batch_size * config.sample_seq_len / (config.num_envs * config.replay_ratio)
    if config.train_every > 1:
        config.train_every = int(config.train_every)
        config.num_iters = 1
    else:
        config.num_iters = int(1 / config.train_every)
        config.train_every = 1
    config.action_dim = gym.make(f"ALE/{config.env_id}-v5").action_space.n
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    return config

if __name__ == "__main__":
    # config = parse_args()
    config = load_config("rltesting/atari_experiments/dreamer-transfer/config.yaml")
    config = process_config(config)
    num_envs = config.num_envs
    env = make_vec_env(make_env(config), num_envs, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='spawn'))
    eval_env = make_vec_env(make_env(config), 1)
    dreamer = DreamerV3(config)
    load_encoder(dreamer, "rltesting/atari_experiments/dreamer-transfer/pretrained_encoder.ckpt")
    load_decoder(dreamer, "rltesting/atari_experiments/dreamer-transfer/pretrained_decoder.ckpt")
    logger = Logger(f"logs/dreamer-v3-transfer/{config.env_id}")

    losses_wm = []
    losses_actor = []
    losses_critic = []
    losses_dict = {}
    obs = env.reset()
    timestep = 0
    for i in range(1, 5_000_000 // config.num_envs + 1):
        timestep += config.num_envs
        if i % 100 == 0:
            print(i)
        action, latent = dreamer.choose_action(torch.tensor(obs).float().to(config.device))
        next_obs, reward, done, info = env.step(to_numpy(action))
        dreamer.process_sample(obs, latent, action, reward, done)
        if True in done:
            indices = np.where(done)[0]
            temp_rew = []
            for index in indices:
                temp_rew.append(info[index]["episode"]["r"])
            logger.add_scalar("rewards/train reward", np.mean(temp_rew))
            logger.write(timestep)
        if i % 1000 == 0 or i == 1:
            eval_reward, obs_traj, frames = eval(dreamer, eval_env, config)
            logger.add_scalar("rewards/eval reward", eval_reward)
            logger.add_video("videos/eval observed", np.concatenate(obs_traj)[np.newaxis, :].astype(np.uint8), fps=120)
            logger.add_video("videos/eval video", np.asarray(frames)[np.newaxis, :].transpose(0, 1, 4, 2, 3), fps=120)
            imagined_rollout = imagine_rollout(dreamer, eval_env, config)
            logger.add_video("videos/imagined rollout", imagined_rollout, fps=120)
            logger.write(timestep)
        if i > 500 and i % config.train_every == 0:
            for _ in range(config.num_iters):
                loss_wm, loss_critic, loss_actor, loss_dict = dreamer.train()
            logger.add_metrics(loss_dict)
            logger.write(timestep)
        obs = next_obs
    