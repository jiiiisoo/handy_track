from isaacgym import gymapi, gymtorch, gymutil
import torch

from DexManipNet.dexmanipnet_oakinkv2 import DexManipNetOakInkV2
from DexManipNet.dexmanipnet_favor import DexManipNetFAVOR
import argparse

from DexManipNet.dexmanip_sh import DexManipSH_RH
from DexManipNet.dexmanip_sh import DexManipSH_LH
from DexManipNet.dexmanip_bih import DexManipBiH

from termcolor import cprint


if __name__ == "__main__":
    # Example usage
    args = gymutil.parse_arguments(
        description="Visualize DexManipNet Dataset",
        headless=True,
        custom_parameters=[
            {
                "name": "--idx",
                "type": int,
                "default": 0,
                "help": "Index of the dataset to visualize",
            },
            {
                "name": "--source",
                "type": str,
                "default": "oakinkv2",
                "help": "Dataset source: [oakinkv2 | favor]",
            },
            {
                "name": "--side",
                "type": str,
                "default": "rh",
                "help": "Side of the dataset to visualize: [rh | lh | bih]",
            },
        ],
    )
    if args.source == "favor":
        data_dir = "data/dexmanipnet/dexmanipnet_favor"
        assert args.side == "rh", "Only rh is supported for favor"
        dataset = DexManipNetFAVOR(data_dir=data_dir, side=args.side)
    elif args.source == "oakinkv2":
        data_dir = "data/dexmanipnet/dexmanipnet_oakinkv2"
        dataset = DexManipNetOakInkV2(data_dir=data_dir, side=args.side)
    else:
        raise ValueError("Invalid source. Choose from [oakinkv2 | favor].")

    if args.side == "rh":
        vis_env = DexManipSH_RH(args, dataset[args.idx])
    elif args.side == "lh":
        vis_env = DexManipSH_LH(args, dataset[args.idx])
    elif args.side == "bih":
        vis_env = DexManipBiH(args, dataset[args.idx])
    else:
        raise ValueError("Invalid side. Choose from [rh | lh | bih].")
    cprint(f"seq_name: {dataset[args.idx]['seq_name']}", "blue")
    if "description" in dataset[args.idx]:
        cprint(f'description: {dataset[args.idx]["description"]}', "red")
    cprint(f'primitive: {dataset[args.idx]["primitive"]}', "green")

    vis_env.play()
