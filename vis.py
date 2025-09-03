import os
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
from torch.distributions.normal import Normal
from collections import deque
import json
from datetime import datetime
import random
"""
This evaluation script reproduces the results from our paper.
For training implementation details, please refer to the paper
and supplementary materials.

Full training code will be released upon paper acceptance.
"""


# Set plotting style
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default')
        print("Warning: Using default matplotlib style")

sns.set_palette("husl")


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Agent class identical to training"""

    def __init__(self, envs):
        super().__init__()
        # Get environment info
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.prod(envs.single_action_space.shape)

        # Shared feature extractor
        self.backbone = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )

        # Value network
        self.critic = nn.Sequential(
            self.backbone,
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # Main policy network
        self.actor_mean = layer_init(nn.Linear(64, action_dim), std=0.01)
        # self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.actor_logstd = nn.Parameter(torch.full((1, action_dim), -1.0))

        # Preference network
        self.preference_mean = layer_init(nn.Linear(64, action_dim), std=0.01)
        self.preference_logstd = nn.Parameter(torch.full((1, action_dim), -1.0))

        # Preference network parameters


    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, deterministic=False, eval_std=None):
        features = self.backbone(x)
        action_mean =  torch.tanh(self.actor_mean(features))
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        pref_mean = torch.tanh(self.preference_mean(features))
        pref_std = torch.exp(self.preference_logstd.expand_as(action_mean)).clamp(min=1e-3, max=2.0)
        combined_mean, combined_std = self.product_of_experts_fusion(
            action_mean, action_std, pref_mean, pref_std
        )
        if deterministic:
            action = combined_mean
            probs = Normal(combined_mean, torch.ones_like(combined_std) * 1e-8)
        elif eval_std is not None:

            fixed_std = torch.full_like(combined_mean, eval_std)
            probs = Normal(combined_mean, fixed_std)
            if action is None:
                action = probs.sample()
        else:
            probs = Normal(combined_mean, combined_std)
            if action is None:
                action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


    def product_of_experts_fusion(self, policy_mean, policy_std, pref_mean, pref_std):
        """Product-of-Experts fusion"""
        policy_precision = 1.0 / (policy_std ** 2 + 1e-8)
        pref_precision = 1.0 / (pref_std ** 2 + 1e-8)

        combined_precision = policy_precision + pref_precision
        combined_mean = (policy_mean * policy_precision + pref_mean * pref_precision) / combined_precision
        combined_std = 1.0 / torch.sqrt(combined_precision + 1e-8)

        combined_std = torch.clamp(combined_std, min=1e-3, max=2.0)
        return combined_mean, combined_std



def make_env(env_id, capture_video=False, video_folder="./evaluation_videos", gamma=0.99):
    """Create environment with EXACT same wrapper order as training"""

    def thunk():

        env = gym.make(env_id, render_mode="rgb_array" if capture_video else None)

        # CRITICAL: Use EXACT same wrapper order as training code
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)  # Records TRUE episode returns
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)  # Applied AFTER RecordEpisodeStatistics
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def evaluate_model(model_path, env_id="HalfCheetah-v4", num_episodes=10, capture_video=True, device="cuda", deterministic=True):  # 🆕 加这个参数
    """Evaluate model with same setup as training"""
    print(f"Starting model evaluation: {model_path}")
    print(f"Number of episodes: {num_episodes}")
    print(f"Recording video: {capture_video}")

    # Create environment with EXACT same wrapper order as training
    video_folder = f"./evaluation_videos_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    env = make_env(env_id, capture_video=capture_video, video_folder=video_folder)()

    # Create agent and load model
    temp_env = gym.make(env_id)
    temp_env = gym.wrappers.FlattenObservation(temp_env)

    class TempEnvs:
        def __init__(self, env):
            self.single_observation_space = env.observation_space
            self.single_action_space = env.action_space

    temp_envs = TempEnvs(temp_env)
    agent = Agent(temp_envs).to(device)

    # Load model weights
    try:
        agent.load_state_dict(torch.load(model_path, map_location=device))
        agent.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Model loading failed: {e}")
        return None

    temp_env.close()

    # Evaluation data collection
    episode_rewards = []  # TRUE episodic returns (from RecordEpisodeStatistics)
    episode_lengths = []
    episode_actions = []
    episode_values = []
    episode_entropies = []
    step_rewards = []  # Normalized step rewards for comparison

    print("\nStarting evaluation...")

    for episode in range(num_episodes):

        frames = [] if capture_video else None


        obs, _ = env.reset()
        obs = torch.FloatTensor(obs).unsqueeze(0).to(device)

        episode_step_rewards = 0  # Sum of normalized step rewards
        episode_length = 0
        actions_this_episode = []
        values_this_episode = []
        entropies_this_episode = []

        done = False

        while not done:

            if capture_video:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            with torch.no_grad():

                action, logprob, entropy, value = agent.get_action_and_value(
                    obs,
                    deterministic=deterministic
                )

                # Record data
                actions_this_episode.append(action.cpu().numpy().flatten())
                values_this_episode.append(value.cpu().item())
                entropies_this_episode.append(entropy.cpu().item())

            obs, reward, terminated, truncated, info = env.step(action.cpu().numpy().flatten())
            obs = torch.FloatTensor(obs).unsqueeze(0).to(device)

            episode_step_rewards += reward  # This is normalized reward
            episode_length += 1
            done = terminated or truncated

            # Check for episode completion info (contains TRUE episodic return)
            if "episode" in info:
                true_episode_return = info["episode"]["r"]
                # Handle numpy array case
                if hasattr(true_episode_return, 'item'):
                    true_episode_return = true_episode_return.item()
                elif isinstance(true_episode_return, (list, tuple, np.ndarray)):
                    true_episode_return = float(true_episode_return[0])
                else:
                    true_episode_return = float(true_episode_return)

                episode_rewards.append(true_episode_return)
                print(f"Episode {episode + 1}/{num_episodes}: True Return={true_episode_return:.2f}, "
                      f"Normalized Sum={episode_step_rewards:.2f}, Length={episode_length}")
                break


        if capture_video and frames:
            try:
                import imageio
                video_path = os.path.join(video_folder, f"episode_{episode:03d}.mp4")
                os.makedirs(video_folder, exist_ok=True)


                imageio.mimsave(video_path, frames, fps=30, format='mp4')
                print(f"  📹 Video saved: {video_path}")
            except Exception as e:
                print(f"  ⚠️ Video save failed with imageio: {e}")

                try:
                    import cv2
                    video_path = os.path.join(video_folder, f"episode_{episode:03d}.avi")
                    os.makedirs(video_folder, exist_ok=True)

                    if frames:
                        height, width, _ = frames[0].shape
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))

                        for frame in frames:
                            # Convert RGB to BGR for opencv
                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            out.write(frame_bgr)

                        out.release()
                        print(f"  📹 Video saved (opencv): {video_path}")
                except Exception as e2:
                    print(f"  ❌ Both imageio and opencv failed: {e2}")
                    print(f"  💡 Suggestion: pip install imageio[ffmpeg] or pip install opencv-python")

        # If no episode info (shouldn't happen with RecordEpisodeStatistics), use step rewards
        if len(episode_rewards) <= episode:
            episode_rewards.append(episode_step_rewards)
            print(
                f"Episode {episode + 1}/{num_episodes}: Step Rewards Sum={episode_step_rewards:.2f}, Length={episode_length}")

        step_rewards.append(episode_step_rewards)
        episode_lengths.append(episode_length)
        episode_actions.append(np.array(actions_this_episode))
        episode_values.append(np.array(values_this_episode))
        episode_entropies.append(np.array(entropies_this_episode))

    env.close()

    # Calculate statistics (rest of the function remains the same)
    results = {
        'episode_rewards': episode_rewards,  # TRUE episodic returns
        'step_rewards': step_rewards,  # Sum of normalized step rewards
        'episode_lengths': episode_lengths,
        'episode_actions': episode_actions,
        'episode_values': episode_values,
        'episode_entropies': episode_entropies,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_step_reward': np.mean(step_rewards),
        'std_step_reward': np.std(step_rewards),
        'mean_length': np.mean(episode_lengths),
        'video_folder': video_folder if capture_video else None
    }

    print(f"\nEvaluation completed!")
    print(f"True Episode Returns - Mean: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Normalized Step Rewards - Mean: {results['mean_step_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Episode Return Range: [{results['min_reward']:.2f}, {results['max_reward']:.2f}]")
    print(f"Mean Episode Length: {results['mean_length']:.1f}")

    if capture_video:
        print(f"Videos saved in: {video_folder}")
        try:

            video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi'))]
            print(f"Generated {len(video_files)} video files:")
            for vf in sorted(video_files):
                print(f"  📹 {vf}")
        except:
            pass

    return results

def plot_evaluation_results(results, save_path="./evaluation_plots"):
    """Plot evaluation results"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('PPO Model Evaluation Results', fontsize=16, fontweight='bold')

    # 1. True episode reward distribution
    axes[0, 0].hist(results['episode_rewards'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(results['mean_reward'], color='red', linestyle='--',
                       label=f'Mean: {results["mean_reward"]:.2f}')
    axes[0, 0].set_xlabel('Episode Returns (True)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Episode Return Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Episode returns over time
    episodes = range(1, len(results['episode_rewards']) + 1)
    axes[0, 1].plot(episodes, results['episode_rewards'], 'o-', color='green',
                    linewidth=2, markersize=6, label='True Episode Returns')
    axes[0, 1].plot(episodes, results['step_rewards'], 's-', color='orange',
                    linewidth=2, markersize=4, label='Normalized Step Rewards')
    axes[0, 1].axhline(results['mean_reward'], color='green', linestyle='--', alpha=0.7)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Return/Reward')
    axes[0, 1].set_title('Episode Returns vs Step Rewards')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Action distribution (first action dimension)
    all_actions = np.concatenate([ep_actions[:, 0] for ep_actions in results['episode_actions']])
    axes[0, 2].hist(all_actions, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 2].set_xlabel('Action Values (Dim 1)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Action Distribution')
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Value function estimates
    all_values = np.concatenate(results['episode_values'])
    axes[1, 0].hist(all_values, bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 0].set_xlabel('Value Estimates')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Value Function Distribution')
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Entropy distribution
    all_entropies = np.concatenate(results['episode_entropies'])
    axes[1, 1].hist(all_entropies, bins=30, alpha=0.7, color='pink', edgecolor='black')
    axes[1, 1].set_xlabel('Policy Entropy')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Policy Entropy Distribution')
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Statistics summary
    axes[1, 2].axis('off')
    stats_text = f"""
Evaluation Statistics Summary

Episode Returns (True):
  Mean: {results['mean_reward']:.2f}
  Std: {results['std_reward']:.2f}
  Min: {results['min_reward']:.2f}
  Max: {results['max_reward']:.2f}

Step Rewards (Normalized):
  Mean: {results['mean_step_reward']:.2f}
  Std: {results['std_step_reward']:.2f}

Episode Stats:
  Mean Length: {results['mean_length']:.1f}
  Total Episodes: {len(results['episode_rewards'])}

Action Stats:
  Action Mean: {np.mean(all_actions):.3f}
  Action Std: {np.std(all_actions):.3f}

Value Stats:
  Value Mean: {np.mean(all_values):.3f}
  Value Std: {np.std(all_values):.3f}
    """
    axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes,
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

    plt.tight_layout()

    # Save plot
    plot_filename = os.path.join(save_path, f'evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Evaluation plots saved to: {plot_filename}")

    plt.show()

    return plot_filename


def save_results_json(results, save_path="./evaluation_results.json"):
    """Save evaluation results to JSON file"""
    json_results = {
        'episode_rewards': results['episode_rewards'],
        'step_rewards': results['step_rewards'],
        'episode_lengths': results['episode_lengths'],
        'mean_reward': float(results['mean_reward']),
        'std_reward': float(results['std_reward']),
        'min_reward': float(results['min_reward']),
        'max_reward': float(results['max_reward']),
        'mean_step_reward': float(results['mean_step_reward']),
        'std_step_reward': float(results['std_step_reward']),
        'mean_length': float(results['mean_length']),
        'video_folder': results['video_folder'],
        'evaluation_time': datetime.now().isoformat()
    }

    with open(save_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"Results saved to: {save_path}")


def main():

    # Configuration
    MODEL_PATH = "PrefPoE.cleanrl_model"
    ENV_ID = "HalfCheetah-v4"
    NUM_EPISODES = 10
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    EVAL_SEEDS = [0, 1, 42, 123, 456, 789, 1234, 2023, 3141, 9999]  # 10个种子
    CAPTURE_VIDEO_SEED = -1

    print("🎲 Multi-Seed PPO Model Evaluation")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        return

    all_results = []
    all_episode_rewards = []


    all_episode_data = []
    all_seed_data = []

    for i, seed in enumerate(EVAL_SEEDS):
        print(f"\n🎲 Seed {i + 1}/{len(EVAL_SEEDS)}: {seed}")
        print("-" * 40)


        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)


        capture_video = (seed == CAPTURE_VIDEO_SEED)




        results = evaluate_model(
            model_path=MODEL_PATH,
            env_id=ENV_ID,
            num_episodes=NUM_EPISODES,
            capture_video=capture_video,
            device=DEVICE,
            deterministic=True
        )

        if results is None:
            print(f"❌ Evaluation failed for seed {seed}")
            continue

        all_results.append(results['mean_reward'])
        all_episode_rewards.extend(results['episode_rewards'])


        seed_data = {
            'seed': seed,
            'mean_reward': results['mean_reward'],
            'std_reward': results['std_reward'],
            'min_reward': results['min_reward'],
            'max_reward': results['max_reward'],
            'episode_rewards': results['episode_rewards'],
            'episode_lengths': results['episode_lengths'],
            'mean_length': results['mean_length']
        }
        all_seed_data.append(seed_data)


        for j, reward in enumerate(results['episode_rewards']):
            episode_data = {
                'seed': seed,
                'episode': j,
                'reward': reward,
                'length': results['episode_lengths'][j],
                'step_reward': results['step_rewards'][j]
            }
            all_episode_data.append(episode_data)

        print(f"Seed {seed} Result: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")


    print("\n" + "=" * 60)
    print("📊 MULTI-SEED EVALUATION SUMMARY")
    print("=" * 60)

    if all_results:
        overall_mean = np.mean(all_results)
        overall_std = np.std(all_results)
        overall_min = np.min(all_results)
        overall_max = np.max(all_results)

        print(f"🎯 Final Performance (across {len(all_results)} seeds):")
        print(f"   Mean: {overall_mean:.2f} ± {overall_std:.2f}")
        print(f"   Range: [{overall_min:.2f}, {overall_max:.2f}]")
        print(f"   Individual Results: {[f'{r:.0f}' for r in all_results]}")


        final_results = {
            'eval_seeds': EVAL_SEEDS[:len(all_results)],
            'seed_means': all_results,
            'overall_mean': float(overall_mean),
            'overall_std': float(overall_std),
            'overall_min': float(overall_min),
            'overall_max': float(overall_max),
            'all_episode_rewards': all_episode_rewards,
            'num_seeds': len(all_results),
            'episodes_per_seed': NUM_EPISODES
        }


        with open('./multi_seed_evaluation_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)


        np.savez('./prefer_evaluation_data.npz',
                 episode_data=all_episode_data,
                 seed_data=all_seed_data,
                 summary=final_results)

        print(f"📊 save to: baseline_evaluation_data.npz")


        plot_multi_seed_results(final_results)

        print(f"\n✅ Multi-seed evaluation completed!")
        print(f"📁 Results saved to: multi_seed_evaluation_results.json")
    else:
        print("❌ No successful evaluations!")


def plot_multi_seed_results(results):

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Multi-Seed Evaluation Results', fontsize=16, fontweight='bold')

    seeds = results['eval_seeds']
    seed_means = results['seed_means']


    axes[0].bar(range(len(seeds)), seed_means, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].axhline(results['overall_mean'], color='red', linestyle='--',
                    label=f'Overall Mean: {results["overall_mean"]:.1f}')
    axes[0].set_xlabel('Seed Index')
    axes[0].set_ylabel('Mean Episode Return')
    axes[0].set_title('Performance Across Seeds')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)


    axes[0].set_xticks(range(len(seeds)))
    axes[0].set_xticklabels([str(s) for s in seeds], rotation=45)


    axes[1].hist(seed_means, bins=min(8, len(seed_means)), alpha=0.7,
                 color='lightgreen', edgecolor='black')
    axes[1].axvline(results['overall_mean'], color='red', linestyle='--',
                    label=f'Mean: {results["overall_mean"]:.1f}')
    axes[1].set_xlabel('Mean Episode Return')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Seed Results')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)


    axes[2].axis('off')
    stats_text = f"""
Multi-Seed Evaluation Summary

Number of Seeds: {results['num_seeds']}
Episodes per Seed: {results['episodes_per_seed']}

Performance Statistics:
  Mean: {results['overall_mean']:.2f}
  Std: {results['overall_std']:.2f}
  Min: {results['overall_min']:.2f}
  Max: {results['overall_max']:.2f}

Range: {results['overall_max'] - results['overall_min']:.1f}
CV: {results['overall_std'] / results['overall_mean'] * 100:.1f}%

Individual Results:
{chr(10).join([f'  Seed {seeds[i]}: {seed_means[i]:.1f}'
               for i in range(len(seeds))])}
    """

    axes[2].text(0.1, 0.9, stats_text, transform=axes[2].transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    plt.tight_layout()


    plot_filename = f'./multi_seed_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📊 Multi-seed plots saved to: {plot_filename}")

    plt.show()


if __name__ == "__main__":
    main()