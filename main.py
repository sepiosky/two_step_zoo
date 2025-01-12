#!/usr/bin/env python3

import argparse
import pprint
import torch

from config import get_configs, load_configs, parse_config_arg
from two_step_zoo import (
    get_two_step_module, get_trainer, get_loaders_from_config,
    get_writer, get_evaluator, get_ood_evaluator
)


parser = argparse.ArgumentParser(description="Two Step Manifold Density Estimator")

parser.add_argument("--dataset", type=str,
    help="Dataset to train on. Required if load-dir not specified.")
parser.add_argument("--gae-model", type=str,
    help="Model for generalized autoencoding. Required if load-dir not specified.")
parser.add_argument("--de-model", type=str,
    help="Model for density estimation. Required if load-dir not specified.")

parser.add_argument("--load-dir", type=str, default="",
    help="Directory to load from.")
parser.add_argument("--load-best-valid-first", action="store_true",
    help="Attempt to load the best_valid checkpoint first.")
parser.add_argument("--load-pretrained-gae", action="store_true",
    help="Load only pretrained gae from resume-dir.")
parser.add_argument("--freeze-pretrained-gae", action="store_true",
    help="Freeze the parameters of the pretrained GAE, i.e. do not train them.")

parser.add_argument("--max-epochs-loaded", type=int,
    help="New maximum shared epochs for loaded model.")
parser.add_argument("--max-epochs-loaded-gae", type=int,
    help="New maximum epochs for loaded GAE model.")
parser.add_argument("--max-epochs-loaded-de", type=int,
    help="New maximum epochs for loaded DE model.")

parser.add_argument("--gae-config", default=[], action="append",
    help="Override gae config entries. Specify as `key=value`.")
parser.add_argument("--de-config", default=[], action="append",
    help="Override de config entries. Specify as `key=value`.")
parser.add_argument("--shared-config", default=[], action="append",
    help="Override shared config entries. Specify as `key=value`.")

parser.add_argument("--only-test", action="store_true",
    help="Only perform a test, no training.")

parser.add_argument("--test-ood", action="store_true",
    help="Perform an OOD test.")

args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"


if args.load_dir:
    gae_cfg, de_cfg, shared_cfg = load_configs(
        args=args,
        density_estimator=args.de_model if args.de_model else None
    )

    if args.load_pretrained_gae:
        # NOTE: When loading a pretrained GAE, we do not expect to load either de_cfg or shared_cfg
        de_cfg = {**de_cfg, **dict(parse_config_arg(kv) for kv in args.de_config)}
        shared_cfg = {**shared_cfg, **dict(parse_config_arg(kv) for kv in args.shared_config)}

else:
    gae_cfg, de_cfg, shared_cfg = get_configs(
        dataset=args.dataset,
        generalized_autoencoder=args.gae_model,
        density_estimator=args.de_model
    )
    gae_cfg = {**gae_cfg, **dict(parse_config_arg(kv) for kv in args.gae_config)}
    de_cfg = {**de_cfg, **dict(parse_config_arg(kv) for kv in args.de_config)}
    de_cfg["data_dim"] = gae_cfg["latent_dim"]
    shared_cfg = {**shared_cfg, **dict(parse_config_arg(kv) for kv in args.shared_config)}


pprint.sorted = lambda x, key=None: x
pp = pprint.PrettyPrinter(indent=4)
print(10*"-" + "-gae_cfg--" + 10*"-")
pp.pprint(gae_cfg)
print(10*"-" + "--de_cfg--" + 10*"-")
pp.pprint(de_cfg)
print(10*"-" + "shared_cfg" + 10*"-")
pp.pprint(shared_cfg)

### TODO train_ds, valid_ds and test_ds explicitly to shared_cfg, de_cfg, gae_cfg
#shared_cfg["train_dset"], shared_cfg["valid_dset"], shared_cfg["test_dset"] = train_dset, test_dset, test_dset
#de_cfg["train_dset"], de_cfg["valid_dset"], de_cfg["test_dset"] = train_dset, test_dset, test_dset
#gae_cfg["train_dset"], gae_cfg["valid_dset"], gae_cfg["test_dset"] = train_dset, test_dset, test_dset

train_loader, valid_loader, test_loader = get_loaders_from_config(shared_cfg)
writer = get_writer(args, gae_cfg=gae_cfg, de_cfg=de_cfg, shared_cfg=shared_cfg)
two_step_module = get_two_step_module(gae_cfg, de_cfg, shared_cfg).to(device)


gae_evaluator = get_evaluator(
    two_step_module.generalized_autoencoder,
    valid_loader=valid_loader, test_loader=test_loader,
    train_loader=train_loader,
    valid_metrics=gae_cfg["valid_metrics"],
    test_metrics=gae_cfg["test_metrics"],
    **gae_cfg.get("metric_kwargs", {}),
)
de_evaluator = get_evaluator(
    two_step_module.density_estimator,
    valid_loader=None, test_loader=None, # Loaders must be updated later by the trainer
    train_loader=train_loader,
    valid_metrics=de_cfg["valid_metrics"],
    test_metrics=de_cfg["test_metrics"],
    **de_cfg.get("metric_kwargs", {}),
)

if args.test_ood or "likelihood_ood_acc" in shared_cfg["test_metrics"]:
    shared_evaluator = get_ood_evaluator(
        two_step_module,
        cfg=shared_cfg,
        include_low_dim=True,
        valid_loader=valid_loader,
        test_loader=test_loader,
        train_loader=train_loader,
        savedir=writer.logdir
    )
else:
    shared_evaluator = get_evaluator(
        two_step_module,
        train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader,
        valid_metrics=shared_cfg["valid_metrics"],
        test_metrics=shared_cfg["test_metrics"],
        **shared_cfg.get("metric_kwargs", {}),
    )


trainer = get_trainer(
    two_step_module=two_step_module,
    writer=writer,
    gae_cfg=gae_cfg,
    de_cfg=de_cfg,
    shared_cfg=shared_cfg,
    train_loader=train_loader,
    valid_loader=valid_loader,
    test_loader=test_loader,
    gae_evaluator=gae_evaluator,
    de_evaluator=de_evaluator,
    shared_evaluator=shared_evaluator,
    load_best_valid_first=args.load_best_valid_first,
    pretrained_gae_path=args.load_dir if args.load_pretrained_gae else "",
    freeze_pretrained_gae=args.freeze_pretrained_gae if args.freeze_pretrained_gae else None,
    only_test=args.only_test
)
trainer.train()
