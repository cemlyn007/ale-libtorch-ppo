#include "rollout.h"
#include "ai/environment/episode_life.h"
#include "ai/environment/episode_observation_recorder.h"
#include "ai/environment/episode_recorder.h"
#include "ai/environment/fire_reset.h"
#include "ai/environment/max_and_skip.h"
#include "ai/environment/noop_reset.h"
#include "ai/environment/truncate_on_episode_return.h"
#include "ai/gae.h"
#include <cassert>

namespace ai::rollout {

Rollout::Rollout(
    std::filesystem::path rom_path, size_t total_environments, size_t horizon,
    size_t max_steps, size_t frame_stack, bool grayscale,
    std::function<ActionResult(const torch::Tensor &)> action_selector,
    float gae_discount, float gae_lambda, const torch::Device &device,
    size_t seed, size_t num_workers, size_t worker_batch_size,
    size_t frame_skip, ale::reward_t max_return,
    std::optional<std::filesystem::path> video_path, bool record_observation)
    : gae_discount_(gae_discount), gae_lambda_(gae_lambda), rom_path_(rom_path),
      buffer_([&] {
        ale::ALEInterface ale;
        ale.loadROM(rom_path);
        auto screen = ale.getScreen();
        std::vector<size_t> observation_shape;
        if (grayscale)
          observation_shape = {frame_stack, screen.height(), screen.width()};
        else
          observation_shape = {frame_stack, 3, screen.height(), screen.width()};
        return ai::buffer::Buffer(total_environments, horizon,
                                  observation_shape,
                                  ale.getMinimalActionSet().size(), device);
      }()),
      total_environments_(total_environments), horizon_(horizon),
      frame_stack_(frame_stack), max_steps_(max_steps), is_terminated_(),
      is_truncated_(), is_episode_start_(),
      game_overs_(total_environments, false),
      episode_returns_(total_environments, 0.0f),
      episode_lengths_(total_environments, 0),
      game_returns_(total_environments, 0.0f),
      game_lengths_(total_environments, 0), action_selector_(action_selector),
      device_(device), environments_(), stop_(), batch_size_(worker_batch_size),
      grayscale_(grayscale), record_observation_(record_observation) {
  if (total_environments_ == 0) {
    throw std::invalid_argument("Total environments must be greater than 0.");
  }
  if (horizon_ == 0) {
    throw std::invalid_argument("Horizon must be greater than 0.");
  }
  if (max_steps_ == 0) {
    throw std::invalid_argument("Max steps must be greater than 0.");
  }
  if (frame_stack_ == 0) {
    throw std::invalid_argument("Frame stack must be greater than 0.");
  }
  if (rom_path_.empty()) {
    throw std::invalid_argument("ROM path must not be empty.");
  }
  if (!std::filesystem::exists(rom_path_)) {
    throw std::invalid_argument(std::string("ROM file does not exist: ") +
                                rom_path_.string());
  }

  environments_.resize(total_environments_);
  screen_buffers_.resize(total_environments_);
  screen_tensor_blobs_.resize(total_environments_);
  std::vector<std::thread> threads;
  for (size_t i = 0; i < total_environments_; ++i) {
    threads.emplace_back([&, i]() {
      auto environment =
          create_environment(i, seed, frame_skip, max_return, video_path);
      auto screen = environment->get_interface().getScreen();
      std::vector<int64_t> observation_shape;
      if (grayscale_) {
        screen_buffers_[i] =
            std::vector<unsigned char>(screen.height() * screen.width());
        observation_shape = {static_cast<int64_t>(screen.height()),
                             static_cast<int64_t>(screen.width())};
      } else {
        screen_buffers_[i] =
            std::vector<unsigned char>(3 * screen.height() * screen.width());
        observation_shape = {3, static_cast<int64_t>(screen.height()),
                             static_cast<int64_t>(screen.width())};
      }
      screen_tensor_blobs_[i] = torch::from_blob(
          screen_buffers_[i].data(), observation_shape, torch::kByte);
      environments_[i] = std::move(environment);
    });
  }
  for (auto &thread : threads)
    thread.join();

  auto screen = environments_[0]->get_interface().getScreen();
  std::vector<int64_t> observation_shape;
  if (grayscale_) {
    observation_shape = {static_cast<int64_t>(screen.height()),
                         static_cast<int64_t>(screen.width())};
  } else {
    observation_shape = {3, static_cast<int64_t>(screen.height()),
                         static_cast<int64_t>(screen.width())};
  }

  auto total = static_cast<int64_t>(total_environments_);
  auto frame = static_cast<int64_t>(frame_stack_);
  auto options = torch::TensorOptions(torch::kFloat32).device(device_);
  std::vector<int64_t> observations_shape({total, frame});
  observations_shape.insert(observations_shape.end(), observation_shape.begin(),
                            observation_shape.end());

  observations_ = torch::zeros(observations_shape, options.dtype(torch::kByte));
  is_terminated_ = torch::zeros({total}, options.dtype(torch::kBool));
  is_truncated_ = torch::zeros({total}, options.dtype(torch::kBool));
  is_episode_start_ = torch::ones({total}, options.dtype(torch::kBool));
  rewards_ = torch::zeros({total}, options);

  is_episode_start_cpu_.resize(total_environments_, true);

  std::cout << "Creating " << num_workers << " worker threads." << std::endl;
  for (size_t i = 0; i < num_workers; ++i) {
    workers_.emplace_back(&Rollout::worker, this);
  }
}

std::unique_ptr<ai::environment::VirtualEnvironment>
Rollout::create_environment(
    size_t i, size_t seed, size_t frame_skip, ale::reward_t max_return,
    const std::optional<std::filesystem::path> &video_path) const {
  std::unique_ptr<ai::environment::VirtualEnvironment> environment =
      std::make_unique<ai::environment::Environment>(rom_path_, max_steps_,
                                                     grayscale_, i + seed);

  // Atari breakout only has two sets of bricks, once the second set is
  // cleared, no more bricks will appear.
  if (max_return > 0.0f)
    environment =
        std::make_unique<ai::environment::TruncateOnEpisodeReturnEnvironment>(
            std::move(environment), max_return);

  if (i == 0 && video_path.has_value()) {
    if (record_observation_)
      environment =
          std::make_unique<ai::environment::EpisodeObservationRecorder>(
              std::move(environment), video_path.value(), grayscale_);
    else
      environment = std::make_unique<ai::environment::EpisodeRecorder>(
          std::move(environment), video_path.value(), false);
  }
  // TODO: Make this configurable.
  environment = std::make_unique<ai::environment::NoopResetEnvironment>(
      std::move(environment), 30, seed + i);
  environment = std::make_unique<ai::environment::MaxAndSkipEnvironment>(
      std::move(environment), frame_skip);
  environment =
      std::make_unique<ai::environment::EpisodeLife>(std::move(environment));
  environment =
      std::make_unique<ai::environment::FireReset>(std::move(environment));
  return environment;
}

Rollout::~Rollout() {
  stop_ = true;
  std::vector<size_t> inputs(total_environments_);
  action_result_.actions.fill_(ale::Action::RANDOM);
  is_episode_start_cpu_.assign(total_environments_, true);
  for (size_t i = 0; i < total_environments_; ++i)
    inputs[i] = i;
  action_queue_.push(inputs);
  for (auto &worker : workers_)
    if (worker.joinable())
      worker.join();
}

void Rollout::update_observations() {
  for (int64_t frame_index = frame_stack_ - 1; frame_index > 0; --frame_index)
    observations_.index_put_(
        {torch::indexing::Slice(), frame_index},
        observations_.index({torch::indexing::Slice(), frame_index - 1}));
  for (size_t i = 0; i < total_environments_; ++i) {
    const auto &frame = screen_tensor_blobs_[i];
    if (is_episode_start_cpu_[i]) {
      observations_.select(0, i).copy_(frame);
    } else {
      observations_.index_put_({static_cast<int64_t>(i), 0}, frame);
    }
  }
}

RolloutResult Rollout::rollout() {
  std::vector<float> episode_returns;
  std::vector<size_t> episode_lengths;
  std::vector<float> game_returns;
  std::vector<size_t> game_lengths;

  for (size_t time_index = 0; time_index < horizon_; time_index++) {

    // Action Selection
    action_result_ = action_selector_(observations_);

    // Step all environments with the selected actions.
    size_t total_steps_increment = 0;
    const auto step_results = step_all();
    for (const auto &result : step_results) {
      int64_t ale_index = result.environment_index;
      if (!is_episode_start_cpu_[ale_index]) {
        rewards_[ale_index] = result.reward;
        is_terminated_[ale_index] = result.terminated;
        is_truncated_[ale_index] = result.truncated;
        game_overs_[ale_index] = result.game_over;
        episode_returns_[ale_index] += result.reward;
        episode_lengths_[ale_index]++;
        game_returns_[ale_index] += result.reward;
        game_lengths_[ale_index]++;
        total_steps_increment++;
      }
    }

    // Add the observations, and the actions that from those observations led
    // to the rewards and terminal state changes.
    buffer_.add(observations_, action_result_.actions, rewards_, is_terminated_,
                is_truncated_, is_episode_start_, action_result_.logits,
                action_result_.values);

    // Get the next observations after taking actions and saving the
    // observations.
    update_observations();

    for (const auto &result : step_results) {
      int64_t ale_index = result.environment_index;
      if (result.terminated || result.truncated) {
        is_episode_start_[ale_index] = true;
        is_episode_start_cpu_[ale_index] = true;
        is_terminated_[ale_index] = false;
        is_truncated_[ale_index] = false;
        current_episode_++;
        episode_returns.push_back(episode_returns_[ale_index]);
        episode_lengths.push_back(episode_lengths_[ale_index]);
        episode_returns_[ale_index] = 0.0;
        episode_lengths_[ale_index] = 0;
        if (game_overs_[ale_index]) {
          game_returns.push_back(game_returns_[ale_index]);
          game_lengths.push_back(game_lengths_[ale_index]);
          game_returns_[ale_index] = 0.0;
          game_lengths_[ale_index] = 0;
        }
      } else if (is_episode_start_cpu_[ale_index]) {
        is_episode_start_[ale_index] = false;
        is_episode_start_cpu_[ale_index] = false;
      }
    }
    total_steps_ += total_steps_increment;
  }
  action_result_ = action_selector_(observations_);
  const auto batch =
      buffer_.get(action_result_.values, gae_discount_, gae_lambda_);
  const Log log{.steps = total_steps_,
                .episodes = current_episode_,
                .episode_returns = episode_returns,
                .episode_lengths = episode_lengths,
                .game_returns = game_returns,
                .game_lengths = game_lengths};
  return {batch, log};
}

std::vector<StepResult> Rollout::step_all() {
  std::vector<size_t> inputs(total_environments_);
  for (size_t i = 0; i < total_environments_; ++i) {
    inputs[i] = i;
  }
  action_queue_.push(inputs);
  return step_queue_.pop(inputs.size());
}

void Rollout::worker() {
  while (!stop_) {
    auto inputs = action_queue_.pop(batch_size_);
    for (const auto &input : inputs) {
      StepResult result = step(input);
      step_queue_.push(result);
    }
  }
}

StepResult Rollout::step(const size_t environment_index) {
  StepResult output;
  output.environment_index = environment_index;
  std::vector<unsigned char> observation;
  if (is_episode_start_cpu_[environment_index]) {
    observation = environments_[environment_index]->reset();
    output.reward = 0.0f;
    output.terminated = false;
    output.truncated = false;
    output.game_over = false;
  } else {
    auto &interface = environments_[environment_index]->get_interface();
    auto action_set = interface.getMinimalActionSet();
    size_t action_index =
        action_result_.actions[environment_index].item<int64_t>();
    if (action_index < 0 || action_index >= action_set.size())
      throw std::out_of_range("Action index out of range for environment " +
                              std::to_string(environment_index));
    auto action = action_set[action_index];
    auto result = environments_[environment_index]->step(action);
    observation = result.observation;
    output.reward = result.reward;
    output.terminated = result.terminated;
    output.truncated = result.truncated;
    output.game_over = result.game_over;
  }
  std::copy(observation.begin(), observation.end(),
            screen_buffers_[environment_index].begin());
  return output;
}
} // namespace ai::rollout