import torch

from .misc import config_to_string


def save_snapshot(file, config, epoch, last_score, best_score, global_step, **kwargs):
    data = {
        "config": config_to_string(config),
        "state_dict": dict(kwargs),
        "training_meta": {
            "epoch": epoch,
            "last_score": last_score,
            "best_score": best_score,
            "global_step": global_step
        }
    }
    torch.save(data, file)


def pre_train_from_snapshots(model, snapshots, modules):
    for snapshot in snapshots:
        if ":" in snapshot:
            module_name, snapshot = snapshot.split(":")
        else:
            module_name = None

        snapshot = torch.load(snapshot, map_location="cpu")
        state_dict = snapshot["state_dict"]

        if module_name is None:
            for module_name in modules:
                if module_name in state_dict:
                    _load_pretraining_dict(getattr(model, module_name), state_dict[module_name])
        else:
            if module_name in modules:
                _load_pretraining_dict(getattr(model, module_name), state_dict[module_name])
            else:
                raise ValueError("Unrecognized network module {}".format(module_name))


def resume_from_snapshot(model, snapshot, modules):
    snapshot = torch.load(snapshot, map_location="cpu")
    state_dict = snapshot["state_dict"]

    for module in modules:
        if module in state_dict:
            _load_pretraining_dict(getattr(model, module), state_dict[module])
        else:
            raise KeyError("The given snapshot does not contain a state_dict for module '{}'".format(module))

    return snapshot


def _load_pretraining_dict(model, state_dict):
    """Load state dictionary from a pre-training snapshot

    This is an even less strict version of `model.load_state_dict(..., False)`, which also ignores parameters from
    `state_dict` that don't have the same shapes as the corresponding ones in `model`. This is useful when loading
    from pre-trained models that are trained on different datasets.

    Parameters
    ----------
    model : torch.nn.Model
        Target model
    state_dict : dict
        Dictionary of model parameters
    """
    model_sd = model.state_dict()

    for k, v in model_sd.items():
        if k in state_dict:
            if v.shape != state_dict[k].shape:
                del state_dict[k]

    model.load_state_dict(state_dict, False)
