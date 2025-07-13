import os
import json
import glob


def get_val_info(data_path):
    """
    Get the validation sequence information from the dataset.
    """
    # Load the validation sequence IDs
    val_seq_id = json.load(open(os.path.join(data_path, "oakinkv2_val_list.json")))

    # Create a list to store the validation sequences
    val_seq = []

    # Iterate through the validation sequence IDs
    for seq in val_seq_id:
        # Get the path to the sequence
        seq_path = glob.glob(os.path.join(data_path, "sequences", seq) + "_*")[0]

        assert os.path.exists(seq_path), f"Path {seq_path} does not exist"

        info = json.load(open(os.path.join(seq_path, "seq_info.json"), "r"))
        if info["type"] == "lh":
            side = "LH"
        elif info["type"] == "rh":
            side = "RH"
        else:
            side = "BiH"
        val_seq.append(
            {
                "dataIndices": "[" + seq + "]",
                "side": side,
                "rolloutLen": info["seq_len"],
                "rolloutBegin": info["start_frame"],
            }
        )

    return val_seq


if __name__ == "__main__":
    # Example usage
    data_path = "data/dexmanipnet/dexmanipnet_oakinkv2"
    val_seq = get_val_info(data_path)
    print(val_seq)
