from collections import defaultdict
from contextlib import contextmanager, nullcontext
import csv
from datetime import datetime
from functools import partial
from pathlib import Path
import shutil
import sys
import time
from typing import Dict, Optional

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from agent import Agent
from collector import Collector
from data import BatchSampler, collate_segments_to_batch, DatasetTraverser, EpisodeCountManager, EpisodeDataset
from envs import SingleProcessEnv, MultiProcessEnv
from make_reconstructions import make_reconstructions_from_batch
from models import ActorCritic, Tokenizer, WorldModel
from utils import configure_optimizer, EpisodeDirManager, plot_counts, set_seed, try_until_no_except


class PhaseTimer:
    """Wall-clock timer for the phases of one epoch.

    CUDA kernels run asynchronously, so we call torch.cuda.synchronize() at the start and end of each phase.
    Otherwise GPU work launched in one phase could be counted in the next one. We only sync at phase
    boundaries (a few times per epoch), so this has no measurable cost.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.times = defaultdict(float)  # phase name -> seconds

    def _sync(self) -> None:
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @contextmanager
    def __call__(self, name: str):
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.times[name] += time.perf_counter() - start


class Trainer:
    def __init__(self, cfg: DictConfig) -> None:
        torch.backends.cuda.matmul.allow_tf32 = True
        if cfg.params.common.seed is None:
            cfg.params.common.seed = int(datetime.now().timestamp()) % 10 ** 5
        set_seed(cfg.params.common.seed)
        try_until_no_except(partial(wandb.init, config=OmegaConf.to_container(cfg, resolve=True), reinit=True, resume=True, **cfg.wandb))

        OmegaConf.resolve(cfg)
        cfg_env_train, cfg_env_test = cfg.env.train, cfg.env.test
        cfg = cfg.params
        self.cfg = cfg
        self.device = torch.device(cfg.common.device)
        self.start_epoch = 1
        self.grad_steps = {}

        self.make_dirs()

        num_actions = self.make_envs(cfg_env_train, cfg_env_test)

        cfg.tokenizer.num_actions = cfg.world_model.num_actions = cfg.actor_critic.model.num_actions = num_actions
        self.agent = Agent(Tokenizer(instantiate(cfg.tokenizer)), WorldModel(instantiate(cfg.world_model)), ActorCritic(instantiate(cfg.actor_critic))).to(self.device)

        self.optimizer_tokenizer = configure_optimizer(self.agent.tokenizer, cfg.training.learning_rate, cfg.training.tokenizer.weight_decay)
        self.optimizer_world_model = configure_optimizer(self.agent.world_model, cfg.training.learning_rate, cfg.training.world_model.weight_decay)
        self.optimizer_actor_critic = torch.optim.Adam(self.agent.actor_critic.parameters(), lr=cfg.training.learning_rate)

        if cfg.initialization.path_to_checkpoint is not None:
            self.agent.load(**cfg.initialization, device=self.device)

        if cfg.common.resume:
            self.load_checkpoint()

    def make_dirs(self) -> None:
        self.ckpt_dir = Path('checkpoints')
        self.media_dir = Path('media')
        self.episode_dir = self.media_dir / 'episodes'
        self.reconstructions_dir = self.media_dir / 'reconstructions'

        if self.cfg.collection.path_to_static_dataset is None:
            self.dataset_dir = self.ckpt_dir / 'dataset'
        else:
            print(f'Using a static dataset, no collection.')
            self.dataset_dir = Path(self.cfg.collection.path_to_static_dataset)
            assert (self.dataset_dir / 'train').is_dir() and (self.dataset_dir / 'test').is_dir()

        if not self.cfg.common.resume:
            config_dir = Path('config')
            config_path = config_dir / 'trainer.yaml'
            config_dir.mkdir(exist_ok=False, parents=False)
            shutil.copy('.hydra/config.yaml', config_path)
            wandb.save(str(config_path))
            shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "src"), dst="./src")
            shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "scripts"), dst="./scripts")
            self.ckpt_dir.mkdir(exist_ok=False, parents=False)
            self.media_dir.mkdir(exist_ok=False, parents=False)
            self.episode_dir.mkdir(exist_ok=False, parents=False)
            torch.save(0, self.ckpt_dir / 'epoch.pt')

    def make_envs(self, cfg_env_train: DictConfig, cfg_env_test: DictConfig) -> int:
        episode_manager_train = EpisodeDirManager(self.episode_dir / 'train', max_num_episodes=self.cfg.collection.train.num_episodes_to_save)
        episode_manager_test = EpisodeDirManager(self.episode_dir / 'test', max_num_episodes=self.cfg.collection.test.num_episodes_to_save)

        def create_env(cfg_env, num_envs):
            env_fn = partial(instantiate, config=cfg_env)
            return MultiProcessEnv(env_fn, num_envs, should_wait_num_envs_ratio=1.0) if num_envs > 1 else SingleProcessEnv(env_fn)

        if self.cfg.training.should:
            train_env = create_env(cfg_env_train, self.cfg.collection.train.num_envs)
            self.train_dataset = EpisodeDataset(directory=self.dataset_dir / 'train', name='train_dataset')
            self.episode_count_manager = EpisodeCountManager(self.train_dataset)
            self.episode_count_manager.register('tokenizer', 'world_model', 'actor_critic')
            self.train_collector = Collector(train_env, self.train_dataset, episode_manager_train, self.episode_count_manager)
            
        if self.cfg.evaluation.should:
            test_env = create_env(cfg_env_test, self.cfg.collection.test.num_envs)
            self.test_dataset = EpisodeDataset(directory=self.dataset_dir / 'test', name='test_dataset')
            self.test_collector = Collector(test_env, self.test_dataset, episode_manager_test)

        assert self.cfg.training.should or self.cfg.evaluation.should
        num_actions = train_env.num_actions if self.cfg.training.should else test_env.num_actions

        return num_actions

    def run(self) -> None:
        run_start_time = time.perf_counter()
        steps_at_start = self.train_dataset.num_steps if self.cfg.training.should else 0

        for epoch in range(self.start_epoch, 1 + self.cfg.common.epochs):

            print(f"\nEpoch {epoch} / {self.cfg.common.epochs}\n")
            start_time = time.time()
            to_log = []
            timer = PhaseTimer(self.device)

            if self.cfg.training.should:
                if self.cfg.collection.path_to_static_dataset is None and epoch <= self.cfg.collection.train.stop_after_epochs:
                    with timer('collect_train'):
                        to_log += self.train_collector.collect(self.agent, epoch, **self.cfg.collection.train.config)
                to_log += self.train_agent(epoch, timer)

            if self.cfg.evaluation.should and (epoch % self.cfg.evaluation.every == 0):
                if self.cfg.collection.path_to_static_dataset is None:
                    with timer('collect_test'):
                        self.test_dataset.clear()
                        to_log += self.test_collector.collect(self.agent, epoch, **self.cfg.collection.test.config, offset_episode_id=(epoch - 1) // self.cfg.evaluation.every  * self.cfg.collection.test.config.num_episodes)
                        self.test_dataset.save_info()
                with timer('eval'):
                    to_log += self.eval_agent(epoch)

            if self.cfg.training.should:
                with timer('checkpoint'):
                    self.save_checkpoint(epoch, save_agent_only=not self.cfg.common.do_checkpoint)
                if epoch == self.cfg.common.epochs:
                    to_log.append({'sampling_counts_tokenizer': wandb.Image(plot_counts(self.episode_count_manager.all_counts['tokenizer']))})
                    to_log.append({'sampling_counts_world_model': wandb.Image(plot_counts(self.episode_count_manager.all_counts['world_model']))})
                    to_log.append({'sampling_counts_actor_critic': wandb.Image(plot_counts(self.episode_count_manager.all_counts['actor_critic']))})

            to_log.append({'duration': (time.time() - start_time) / 3600})
            to_log.append(self.log_timing(epoch, timer, time.time() - start_time, time.perf_counter() - run_start_time, steps_at_start))

            for d in to_log:
                wandb.log({'epoch': epoch, **d})

        self.finish()

    def log_timing(self, epoch: int, timer: PhaseTimer, epoch_seconds: float, total_seconds: float, steps_at_start: int) -> Dict[str, float]:
        """Print a per-phase timing table, append it to timing.csv (in the run folder), and return it for wandb."""
        num_env_steps = self.train_dataset.num_steps if self.cfg.training.should else 0
        new_env_steps = num_env_steps - steps_at_start

        row = {
            'epoch': epoch,
            **{f'time/{k}': v for k, v in timer.times.items()},
            'time/epoch': epoch_seconds,
            'time/total': total_seconds,
            'time/num_env_steps': num_env_steps,
            # same definition as the paper's Table 1: environment frames collected / training duration
            'time/fps': new_env_steps / total_seconds if total_seconds > 0 else 0.0,
        }
        for name in ('tokenizer', 'world_model', 'actor_critic'):
            if name in self.grad_steps and self.grad_steps[name] > 0:
                row[f'time/{name}_ms_per_step'] = 1000 * timer.times[f'train_{name}'] / self.grad_steps[name]

        print(f"\n[timing] epoch {epoch}: {epoch_seconds:.1f}s this epoch, {total_seconds / 3600:.2f}h total, fps={row['time/fps']:.1f}")
        for k, v in timer.times.items():
            ms_key = 'time/' + k.replace('train_', '', 1) + '_ms_per_step'
            extra = f'  ({row[ms_key]:.1f} ms/step)' if ms_key in row else ''
            print(f'[timing]   {k:<22s} {v:9.1f}s  {100 * v / epoch_seconds:5.1f}%{extra}')

        csv_path = Path('timing.csv')
        fieldnames = ['epoch', 'time/collect_train', 'time/train_tokenizer', 'time/train_world_model', 'time/train_actor_critic',
                      'time/collect_test', 'time/eval', 'time/checkpoint', 'time/epoch', 'time/total', 'time/num_env_steps', 'time/fps',
                      'time/tokenizer_ms_per_step', 'time/world_model_ms_per_step', 'time/actor_critic_ms_per_step']
        is_new = not csv_path.exists()
        with csv_path.open('a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            if is_new:
                writer.writeheader()
            writer.writerow(row)

        return row

    def train_agent(self, epoch: int, timer: PhaseTimer) -> None:
        self.agent.train()
        self.agent.zero_grad()

        to_log_tokenizer, to_log_world_model, to_log_actor_critic = {}, {}, {}

        cfg_tokenizer = DictConfig(self.cfg.training.tokenizer)
        cfg_world_model = DictConfig(self.cfg.training.world_model)
        cfg_actor_critic = self.cfg.training.actor_critic

        if epoch == 1 and cfg_tokenizer.steps_first_epoch is not None:
            cfg_tokenizer.steps_per_epoch = cfg_tokenizer.steps_first_epoch

        if epoch == 1 and cfg_world_model.steps_first_epoch is not None:
            cfg_world_model.steps_per_epoch = cfg_world_model.steps_first_epoch

        self.grad_steps = {}  # component name -> number of gradient steps this epoch (for ms/step timing)

        if epoch > cfg_tokenizer.start_after_epochs:
            with timer('train_tokenizer'):
                to_log_tokenizer = self.train_component(
                    self.agent.tokenizer,
                    self.optimizer_tokenizer,
                    can_sample_beyond_end=True,
                    should_update_counts=(epoch > 1),
                    use_mixed_precision=True,
                    **cfg_tokenizer
                )
            self.grad_steps['tokenizer'] = cfg_tokenizer.steps_per_epoch
        self.agent.tokenizer.eval()

        if epoch > cfg_world_model.start_after_epochs:
            with timer('train_world_model'):
                to_log_world_model = self.train_component(
                    self.agent.world_model,
                    self.optimizer_world_model,
                    can_sample_beyond_end=True,
                    tokenizer=self.agent.tokenizer,
                    should_update_counts=(epoch > 1),
                    use_mixed_precision=True,
                    **cfg_world_model
                )
            self.grad_steps['world_model'] = cfg_world_model.steps_per_epoch
        self.agent.world_model.eval()

        # Actor-critic training = imagination rollouts in the world model (expected bottleneck: autoregressive Delta-token sampling)
        if epoch > cfg_actor_critic.start_after_epochs:
            with timer('train_actor_critic'):
                to_log_actor_critic = self.train_component(
                    self.agent.actor_critic,
                    self.optimizer_actor_critic,
                    can_sample_beyond_end=False,
                    should_update_counts=(epoch > 1),
                    use_mixed_precision=False,
                    tokenizer=self.agent.tokenizer,
                    world_model=self.agent.world_model,
                    **cfg_actor_critic
                )
            self.grad_steps['actor_critic'] = cfg_actor_critic.steps_per_epoch
        self.agent.actor_critic.eval()

        return to_log_tokenizer, to_log_world_model, to_log_actor_critic

    def train_component(
        self,
        component: nn.Module,
        optimizer: torch.optim.Optimizer,
        steps_per_epoch: int,
        batch_num_samples: int,
        grad_acc_steps: int,
        max_grad_norm: Optional[float],
        priority_alpha: float,
        sequence_length: int,
        can_sample_beyond_end: bool,
        should_update_counts: bool,
        use_mixed_precision: bool,
        **kwargs_loss
        ) -> Dict[str, float]:

        batch_sampler = BatchSampler(self.train_dataset, steps_per_epoch * grad_acc_steps, batch_num_samples, sequence_length, can_sample_beyond_end) 
        batch_sampler.probabilities = self.episode_count_manager.compute_probabilities(key=str(component), alpha=priority_alpha)
        use_cuda = self.device.type == 'cuda'
        loader = DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            num_workers=self.cfg.training.get('num_workers', 4),
            collate_fn=collate_segments_to_batch,
            pin_memory=use_cuda,  # pinned memory only helps (and only works) for CPU -> GPU copies
            pin_memory_device=str(self.device) if use_cuda else ''
        ) # DataLoader preloads batches, so counts are slightly lagging behind.

        # bf16 autocast is meant for GPU; on CPU (local debugging, e.g. a Mac) it is slow or unsupported, so use fp32 there
        use_mixed_precision = use_mixed_precision and use_cuda

        to_log = defaultdict(float)
        optimizer.zero_grad()

        with tqdm(total=steps_per_epoch, desc=f'Training {component}', file=sys.stdout) as pbar:
            for i, batch in enumerate(loader):
                
                batch = batch.to(self.device)
                # GPU: unchanged from the original code. CPU: no autocast (CPU autocast only accepts bf16)
                autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16 if use_mixed_precision else torch.float32) if use_cuda else nullcontext()
                with autocast_ctx:
                    losses, metrics = component.compute_loss(batch, **kwargs_loss)
                loss = losses.loss_total / grad_acc_steps
                loss.backward()

                for k, v in {**losses.intermediate_losses, **metrics}.items():
                    to_log[f'{component}/train/{k}'] += v / (grad_acc_steps * steps_per_epoch)
                
                if should_update_counts:
                    for segment_id in batch.segment_ids:
                        self.episode_count_manager.increment_episode_count(key=str(component), episode_id=segment_id.episode_id)
                    batch_sampler.probabilities = self.episode_count_manager.compute_probabilities(key=str(component), alpha=priority_alpha)
                
                if (i + 1) % grad_acc_steps == 0:
                    if max_grad_norm is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(component.parameters(), max_grad_norm)
                        to_log[f'{component}/train/grad_norm'] += grad_norm / steps_per_epoch

                    optimizer.step()
                    optimizer.zero_grad()
                    
                    pbar.update(1)

        return to_log

    @torch.no_grad()
    def eval_agent(self, epoch: int) -> None:
        self.agent.eval()

        to_log_tokenizer, to_log_world_model = {}, {}

        cfg_tokenizer = self.cfg.evaluation.tokenizer
        cfg_world_model = self.cfg.evaluation.world_model

        if epoch > cfg_tokenizer.start_after_epochs:
            to_log_tokenizer = self.eval_component(self.agent.tokenizer, cfg_tokenizer.batch_num_samples, sequence_length=self.cfg.training.tokenizer.sequence_length)

        if epoch > cfg_world_model.start_after_epochs:
            to_log_world_model = self.eval_component(self.agent.world_model, cfg_world_model.batch_num_samples, sequence_length=self.cfg.training.world_model.sequence_length, tokenizer=self.agent.tokenizer)

        if cfg_tokenizer.save_reconstructions:
            loader = DataLoader(
                self.test_dataset,
                batch_sampler=BatchSampler(self.test_dataset, 1, 64, self.cfg.training.tokenizer.sequence_length, can_sample_beyond_end=True),
                collate_fn=collate_segments_to_batch
            )
            batch = next(iter(loader)).to(self.device)
            make_reconstructions_from_batch(batch, save_dir=self.reconstructions_dir, epoch=epoch, tokenizer=self.agent.tokenizer, should_overwrite=True)

        return to_log_tokenizer, to_log_world_model

    @torch.no_grad()
    def eval_component(self, component: nn.Module, batch_num_samples: int, sequence_length: int, **kwargs_loss) -> Dict[str, float]:
        to_log = defaultdict(float)
        loader = DatasetTraverser(self.test_dataset, batch_num_samples, sequence_length)

        for batch in tqdm(loader, desc=f"Evaluating {component}", file=sys.stdout):
            
            batch = batch.to(self.device)
            losses, metrics = component.compute_loss(batch, **kwargs_loss)

            for k, v in {**losses.intermediate_losses, **metrics}.items():
                to_log[f'{component}/eval/{k}'] += v
            
        to_log = {k: v / len(loader) for k, v in to_log.items()}
        
        return to_log

    def _save_checkpoint(self, epoch: int, save_agent_only: bool) -> None:
        torch.save(self.agent.state_dict(), self.ckpt_dir / 'last.pt')

        if not save_agent_only:
            torch.save(epoch, self.ckpt_dir / 'epoch.pt')
            torch.save({"optimizer_tokenizer": self.optimizer_tokenizer.state_dict(),
                        "optimizer_world_model": self.optimizer_world_model.state_dict(),
                        "optimizer_actor_critic": self.optimizer_actor_critic.state_dict()},
                        self.ckpt_dir / 'optimizer.pt')
            self.train_dataset.save_info()
            self.episode_count_manager.save(self.ckpt_dir / 'episode_count.pt')

    def save_checkpoint(self, epoch: int, save_agent_only: bool) -> None:
        tmp_checkpoint_dir = Path('checkpoints_tmp')
        shutil.copytree(src=self.ckpt_dir, dst=tmp_checkpoint_dir, ignore=shutil.ignore_patterns('dataset'))
        self._save_checkpoint(epoch, save_agent_only)
        shutil.rmtree(tmp_checkpoint_dir)

    def load_checkpoint(self) -> None:
        assert self.ckpt_dir.is_dir()
        epoch = torch.load(self.ckpt_dir / 'epoch.pt')
        if epoch == 0:
            return 
        self.start_epoch = epoch + 1
        self.agent.load(self.ckpt_dir / 'last.pt', device=self.device)
        ckpt_opt = torch.load(self.ckpt_dir / 'optimizer.pt', map_location=self.device)
        self.optimizer_tokenizer.load_state_dict(ckpt_opt['optimizer_tokenizer'])
        self.optimizer_world_model.load_state_dict(ckpt_opt['optimizer_world_model'])
        self.optimizer_actor_critic.load_state_dict(ckpt_opt['optimizer_actor_critic'])
        self.episode_count_manager.load(self.ckpt_dir / 'episode_count.pt')
        print(f'Successfully loaded model and optimizer from {self.ckpt_dir.absolute()}.')

    def finish(self) -> None:
        wandb.finish()
