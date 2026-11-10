from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from sage_net.backbone import RemoteCLIP
from sage_net.data import EpisodeSampler, scan_image_folder, subsample, to_items
from sage_net.engine import (VARIANTS, Config, LinearProbe, build_sage, load_adapter, make_loader,
                             predict_probe, predict_sage, predict_zero_shot, save_adapter,
                             set_seed, summarize, train_probe, train_sage)
from sage_net.model import count_trainable
from sage_net.prompts import encode_prompts, normalize_class_name


def add_common(p: argparse.ArgumentParser):
    p.add_argument("--checkpoint", required=True, help="RemoteCLIP weights (.pt)")
    p.add_argument("--arch", default="ViT-B-32", choices=["ViT-B-32", "ViT-L-14"])
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--variant", default="full", choices=list(VARIANTS))
    p.add_argument("--method", default="sage", choices=["sage", "linear_probe", "zero_shot"])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-every", type=int, default=0, help="print training loss every N epochs")

    g = p.add_argument_group("config overrides")
    g.add_argument("--num-tokens", type=int)
    g.add_argument("--prompt-set", choices=["classname", "identity", "identity_object",
                                            "identity_geometry", "full"])
    g.add_argument("--prompt-fusion", choices=["gated", "equal"])
    g.add_argument("--geometry", choices=["none", "centroid", "centroid_area",
                                          "centroid_area_aspect", "full"])
    g.add_argument("--no-inject-geometry", dest="inject_geometry", action="store_const", const=False)
    g.add_argument("--no-geometry-bias", dest="geometry_bias", action="store_const", const=False)
    g.add_argument("--lambda-align", type=float)
    g.add_argument("--temperature", type=float)
    g.add_argument("--epochs", type=int)
    g.add_argument("--lr", type=float)
    g.add_argument("--weight-decay", type=float)
    g.add_argument("--batch-size", type=int)
    g.add_argument("--num-workers", type=int)
    g.add_argument("--prompt-overrides")


OVERRIDE_KEYS = ("num_tokens", "prompt_set", "prompt_fusion", "geometry", "inject_geometry",
                 "geometry_bias", "lambda_align", "temperature", "epochs", "lr", "weight_decay",
                 "batch_size", "num_workers", "prompt_overrides")


def make_config(args) -> Config:
    overrides = dict(VARIANTS[args.variant])
    overrides.update({k: getattr(args, k) for k in OVERRIDE_KEYS if getattr(args, k) is not None})
    config_path = args.config if Path(args.config).is_file() else None
    return Config.load(config_path, **overrides)


def write_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2))


def run_fewshot(args, cfg: Config, encoder: RemoteCLIP, out_dir: Path):
    classes = scan_image_folder(args.data_root)
    names = sorted(classes)
    print(f"{len(names)} classes, {sum(map(len, classes.values()))} images in {args.data_root}")

    prompt_bank = None
    if args.method == "sage":
        prompt_bank = encode_prompts(encoder, names, cfg.prompt_set, cfg.prompt_overrides)
        index = {n: i for i, n in enumerate(names)}

    episodes, per_seed = [], {}
    dump = {"classes": [], "pred": [], "target": [], "features": [], "prompt_weights": []}
    for seed in args.seeds:
        sampler = EpisodeSampler(classes, args.ways, args.shots, args.queries, seed=seed)
        accs = []
        for ep in range(args.episodes):
            chosen, support, query = sampler.sample()
            ep_seed = seed * 100_000 + ep
            set_seed(ep_seed)
            query_loader = make_loader(query, cfg, train=False)
            start = time.time()

            if args.method == "zero_shot":
                result = predict_zero_shot(encoder, chosen, query_loader, cfg)
            else:
                train_loader = make_loader(support, cfg, train=True, seed=ep_seed, cache=True)
                if args.method == "sage":
                    emb = prompt_bank[[index[c] for c in chosen]]
                    model = build_sage(encoder, emb, cfg)
                    train_sage(model, train_loader, cfg, log_every=args.log_every)
                    result = predict_sage(model, query_loader, cfg, keep_features=args.save_features)
                else:
                    model = LinearProbe(encoder, len(chosen))
                    train_probe(model, train_loader, cfg)
                    result = predict_probe(model, query_loader, cfg)

            accs.append(result["accuracy"])
            episodes.append({"seed": seed, "episode": ep, "classes": chosen,
                             "accuracy": result["accuracy"]})
            dump["classes"].append(chosen)
            dump["pred"].append(result["pred"])
            dump["target"].append(result["target"])
            if "features" in result:
                dump["features"].append(result["features"])
            if "prompt_weights" in result:
                dump["prompt_weights"].append(result["prompt_weights"])
            print(f"seed {seed} episode {ep + 1:4d}/{args.episodes}  acc {100 * result['accuracy']:6.2f}  "
                  f"running {100 * np.mean(accs):6.2f}  ({time.time() - start:.1f}s)", flush=True)
        per_seed[seed] = summarize(accs)
        print(f"seed {seed}: {per_seed[seed]['mean']:.2f} +- {per_seed[seed]['ci95']:.2f} (95% CI)")

    seed_means = [s["mean"] / 100 for s in per_seed.values()]
    summary = {
        "over_seeds": summarize(seed_means),
        "over_episodes": summarize([e["accuracy"] for e in episodes]),
        "per_seed": per_seed,
    }
    write_json(out_dir / "results.json", {
        "mode": "fewshot", "data_root": str(args.data_root), "method": args.method,
        "variant": args.variant, "ways": args.ways, "shots": args.shots, "queries": args.queries,
        "episodes": args.episodes, "seeds": args.seeds, "config": asdict(cfg),
        "summary": summary, "episode_results": episodes,
    })

    arrays = {
        "classes": np.array(dump["classes"]),
        "pred": np.stack(dump["pred"]),
        "target": np.stack(dump["target"]),
    }
    if dump["features"]:
        arrays["features"] = np.stack(dump["features"])
    if dump["prompt_weights"]:
        arrays["prompt_weights"] = np.stack(dump["prompt_weights"])
    np.savez_compressed(out_dir / "predictions.npz", **arrays)

    s = summary["over_seeds"]
    e = summary["over_episodes"]
    print(f"\n{args.ways}-way {args.shots}-shot: {s['mean']:.2f} +- {s['std']:.2f} (std over "
          f"{s['n']} seeds); episode-level 95% CI +- {e['ci95']:.2f} over {e['n']} episodes")


def run_transfer(args, cfg: Config, encoder: RemoteCLIP, out_dir: Path):
    mapping_all = json.loads(Path(args.mapping).read_text())
    mapping = {normalize_class_name(t): normalize_class_name(s)
               for t, s in mapping_all[args.target].items()}

    source = scan_image_folder(args.source_root)
    target = scan_image_folder(args.target_root)
    missing = [s for s in mapping.values() if s not in source] + [t for t in mapping if t not in target]
    if missing:
        raise KeyError(f"classes in the mapping not found on disk: {missing}")

    source_names = sorted(set(mapping.values()))
    label_of = {name: i for i, name in enumerate(source_names)}
    test_items = [(p, label_of[mapping[t]]) for t in sorted(mapping) for p in target[t]]
    test_loader = make_loader(test_items, cfg, train=False)
    print(f"{len(source_names)} matched classes; {len(test_items)} target test images")

    prompt_bank = None
    if args.method == "sage" and not args.adapter:
        prompt_bank = encode_prompts(encoder, source_names, cfg.prompt_set, cfg.prompt_overrides)

    runs = []
    for seed in args.seeds:
        set_seed(seed)
        if args.method == "zero_shot":
            result = predict_zero_shot(encoder, source_names, test_loader, cfg)
        elif args.adapter:
            model, names, cfg_saved = load_adapter(args.adapter, encoder)
            if names != source_names:
                raise ValueError("adapter classes do not match the mapping")
            result = predict_sage(model, test_loader, cfg_saved, keep_features=args.save_features)
        else:
            train_set = subsample({n: source[n] for n in source_names}, args.shots, seed)
            train_loader = make_loader(to_items(train_set, source_names), cfg, train=True, seed=seed,
                                       cache=args.shots > 0)
            if args.method == "sage":
                model = build_sage(encoder, prompt_bank, cfg)
                print(f"seed {seed}: training {count_trainable(model):,} parameters "
                      f"on {len(train_loader.dataset)} source images")
                history = train_sage(model, train_loader, cfg, log_every=args.log_every or 10)
                save_adapter(out_dir / f"adapter_seed{seed}.pt", model, source_names, cfg,
                             encoder.arch, {"history": history, "seed": seed, "mapping": mapping})
                result = predict_sage(model, test_loader, cfg, keep_features=args.save_features)
            else:
                model = LinearProbe(encoder, len(source_names))
                train_probe(model, train_loader, cfg)
                result = predict_probe(model, test_loader, cfg)

        per_class = {}
        for t in sorted(mapping):
            mask = result["target"] == label_of[mapping[t]]
            per_class[t] = float((result["pred"][mask] == result["target"][mask]).mean())
        runs.append({"seed": seed, "accuracy": result["accuracy"], "per_class": per_class})
        np.savez_compressed(out_dir / f"predictions_seed{seed}.npz",
                            pred=result["pred"], target=result["target"],
                            source_classes=np.array(source_names),
                            **({"features": result["features"]} if "features" in result else {}))
        print(f"seed {seed}: target accuracy {100 * result['accuracy']:.2f}")

    summary = summarize([r["accuracy"] for r in runs])
    write_json(out_dir / "results.json", {
        "mode": "transfer", "target": args.target, "method": args.method, "variant": args.variant,
        "shots_per_source_class": args.shots, "mapping": mapping, "config": asdict(cfg),
        "summary": summary, "runs": runs,
    })
    print(f"\nNWPU -> {args.target}: {summary['mean']:.2f} +- {summary['std']:.2f} over {summary['n']} seeds")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    fs = sub.add_parser("fewshot", help="N-way K-shot episodes")
    fs.add_argument("--data-root", required=True)
    fs.add_argument("--ways", type=int, default=5)
    fs.add_argument("--shots", type=int, default=1)
    fs.add_argument("--queries", type=int, default=15)
    fs.add_argument("--episodes", type=int, default=100, help="episodes per seed")
    fs.add_argument("--save-features", action="store_true", help="store pooled query features")
    add_common(fs)

    tr = sub.add_parser("transfer", help="NWPU-RESISC45 -> AID / UC-Merced")
    tr.add_argument("--source-root", required=True)
    tr.add_argument("--target-root", required=True)
    tr.add_argument("--target", required=True, choices=["aid", "ucm"])
    tr.add_argument("--mapping", default="configs/transfer_mappings.json")
    tr.add_argument("--shots", type=int, default=16, help="images per source class, 0 = all")
    tr.add_argument("--adapter", help="evaluate a saved adapter instead of training")
    tr.add_argument("--save-features", action="store_true")
    add_common(tr)

    args = parser.parse_args()
    cfg = make_config(args)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder = RemoteCLIP(args.checkpoint, args.arch).to(args.device)
    print(f"RemoteCLIP {args.arch} on {args.device}; method={args.method} variant={args.variant}")
    print(json.dumps(asdict(cfg)))

    if args.mode == "fewshot":
        run_fewshot(args, cfg, encoder, out_dir)
    else:
        run_transfer(args, cfg, encoder, out_dir)


if __name__ == "__main__":
    main()
