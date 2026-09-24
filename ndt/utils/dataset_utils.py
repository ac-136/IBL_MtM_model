import numpy as np
from scipy.sparse import csr_array
from datasets import load_dataset, concatenate_datasets
import os
from tqdm import tqdm


def get_binned_spikes_from_sparse(spikes_sparse_data_list, spikes_sparse_indices_list, spikes_sparse_indptr_list, spikes_sparse_shape_list):
    sparse_binned_spikes = [csr_array((spikes_sparse_data_list[i], spikes_sparse_indices_list[i], spikes_sparse_indptr_list[i]), shape=spikes_sparse_shape_list[i]) for i in range(len(spikes_sparse_data_list))]

    binned_spikes = np.array([csr_matrix.toarray() for csr_matrix in sparse_binned_spikes])

    return binned_spikes

def session_data_dir(data_path, eid):
    """Path to one session's data/ dir, accepting either <eid> or <eid>_aligned.

    Session directories are currently named <eid>_aligned but are moving to a
    plain <eid>. Callers pass whichever form they have and this finds the one
    that exists, so no caller needs to know the layout.
    """
    bare = eid.removesuffix("_aligned")
    for name in (bare, bare + "_aligned"):
        path = os.path.join(data_path, name, "data")
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(f"No data/ directory for session {bare!r} under {data_path}")


SPLITS = ("train", "val", "test")


def load_locally(data_path, eid):
    """Load one session from <data_path>/<eid>/data/{train,val,test}.parquet.

    All three splits are required.
    """
    data_dir = session_data_dir(data_path, eid)

    split_files = {s: os.path.join(data_dir, f"{s}.parquet") for s in SPLITS}
    missing = [s for s, path in split_files.items() if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            f"Session {eid!r} is missing {', '.join(f'{s}.parquet' for s in missing)} "
            f"in {data_dir}"
        )

    dataset = load_dataset("parquet", data_files=split_files)

    print("Load locally path: ", data_dir)
    print("Available splits: ", list(dataset.keys()))

    return dataset


def load_ibl_dataset_locally(
                     data_type="just_spikes",
                     base_path='/work/hdd/beml/ac136_codex_test',
                     eid=None,               # a single session; otherwise use train_session_eid
                     num_sessions=5,         # cap on how many sessions to load
                     train_session_eid=[],
                     test_session_eid=[],    # sessions to hold out of training
                     data_root=None):
    """Load sessions from disk, using each one's predefined train/val/test splits.

    Returns (train, val, test, meta_data), where the splits are concatenated
    across sessions and meta_data carries num_neurons, num_sessions and eids.
    """

    print("\nload_ibl_dataset_locally parameters: ")
    
    print(f"data_type: ", data_type)
    print(f"eid: ", eid)
    print(f"num_sessions: ", num_sessions)
    print(f"train_session_eid: ", train_session_eid)
    print(f"test_session_eid: ", test_session_eid)

    ### HANDLE DIFF DATASET LOADING BETWEEN SPIKES ONLY AND OG DATASETS ###
    # Keep ``data_type`` as the semantic dataset format while allowing callers
    # to load that format from a directory whose basename is different.  This
    # matters for fine-tuning because ``data_type`` also controls checkpoint
    # handling and therefore cannot safely be inferred from the directory name.
    data_path = os.path.abspath(os.path.expanduser(data_root)) if data_root else os.path.join(base_path, data_type)
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Dataset root not found: {data_path}")

    print(f"data_path: {data_path}")
    print()

    dataset_name_list = os.listdir(data_path) # all aligned

    print("Total session-wise datasets found: ", len(dataset_name_list))
    print("Eid is none test: ", eid)

    test_session_eid_dir = []
    train_session_eid_dir = []
    if eid is not None: # eval only
        # Either naming works; session_data_dir resolves it and raises a clear
        # error if neither directory exists.
        train_session_eid_dir = [eid]
        user_datasets = [eid]

    # print(train_session_eid_dir)
    # print(user_datasets)
    # print()


    if len(test_session_eid) > 0:
        test_session_eid_dir = list(test_session_eid)
        print("Test session-wise datasets found: ", len(test_session_eid_dir))
        excluded = {e.removesuffix("_aligned") for e in test_session_eid}
        train_session_eid_dir = [e for e in user_datasets
                                 if e.removesuffix("_aligned") not in excluded]
        print("Train session-wise datasets found: ", len(train_session_eid_dir))
        train_session_eid_dir = train_session_eid_dir[:num_sessions - len(test_session_eid)]
        print("Number of training sesssion datasets to be used: ", len(train_session_eid_dir))
    else:
        if len(train_session_eid) > 0:
            train_session_eid_dir = list(train_session_eid)
            # M1: an explicit list longer than num_sessions used to be truncated
            # silently by the [:num_sessions] slice below, so a caller asking for N
            # sessions could train on fewer with nothing said. Refuse instead.
            if len(train_session_eid_dir) > num_sessions:
                raise ValueError(
                    f"{len(train_session_eid_dir)} sessions requested but num_sessions="
                    f"{num_sessions}; refusing to truncate the session list silently.")
        else:
            train_session_eid_dir = user_datasets

    assert len(train_session_eid_dir) > 0, "No training datasets found"

    print("Loading train dataset sessions using their predefined train/val/test splits...")
    session_train_datasets = []
    session_val_datasets = []
    session_test_datasets = []

    num_neuron_set = set()
    eids_set = set()
    # M1: ordered (eid, num_neurons) in load order. num_neurons stays a set-derived
    # list exactly as before, so single-session model construction is untouched;
    # this is the per-session mapping multi-session code needs and that a set loses.
    session_num_neurons = []
    # The `use_re` filter was removed with the eid-list helpers: it is False on
    # every code path here, so target_eids/test_re_eids were read and discarded.

    print(train_session_eid_dir)
    print(train_session_eid_dir[:num_sessions])    
    for dataset_eid in tqdm(train_session_eid_dir[:num_sessions]):
        try:
            print("Dataset id: ", dataset_eid)
            dataset_eid = dataset_eid.split('/')[-1]
            print("Dataset id: ", dataset_eid)
            
            session_dataset = load_locally(data_path=data_path, eid=dataset_eid)

            print("Results of load_locally: ", session_dataset)
            print()

            session_train_datasets.append(session_dataset["train"])
            session_val_datasets.append(session_dataset["val"])

            session_test_datasets.append(session_dataset["test"])

            binned_spikes_data = get_binned_spikes_from_sparse([session_dataset["train"]["spikes_sparse_data"][0]], 
                                                                [session_dataset["train"]["spikes_sparse_indices"][0]],
                                                                [session_dataset["train"]["spikes_sparse_indptr"][0]],
                                                                [session_dataset["train"]["spikes_sparse_shape"][0]])
            
            # print("Binned spikes data: ")
            # print(binned_spikes_data)
            # print(binned_spikes_data.shape)

            num_neuron_set.add(binned_spikes_data.shape[2])
            session_num_neurons.append([dataset_eid.removesuffix("_aligned"),
                                        int(binned_spikes_data.shape[2])])

            # The bare eid is the session's canonical identity: it is what the
            # parquet `eid` column holds and what the session embedding keys on.
            eids_set.add(dataset_eid.removesuffix("_aligned"))
        except Exception as e:
            # M1: fail loudly. The previous `continue` let a scale-N run train on
            # N-1 sessions with nothing but a print, and could leave a session's
            # train/val splits appended when a later step for it failed.
            raise RuntimeError(f"failed to load requested session {dataset_eid}: "
                               f"{type(e).__name__}: {e}") from e
    print("session eid used: ", eids_set)
    print("Total number of session: ", len(eids_set))
    if len(session_train_datasets) == 0:
        raise ValueError("No training datasets were loaded successfully.")
    if len(session_val_datasets) == 0:
        raise ValueError("No validation datasets were loaded successfully.")
    train_dataset = concatenate_datasets(session_train_datasets)
    val_dataset = concatenate_datasets(session_val_datasets)
    test_dataset = concatenate_datasets(session_test_datasets)
    print("Train dataset size: ", len(train_dataset))
    print("Val dataset size: ", len(val_dataset))
    print("Test dataset size: ", len(test_dataset))

    num_neuron_set = list(num_neuron_set)
    meta_data = {
        "num_neurons": num_neuron_set,
        "num_sessions": len(eids_set),
        "eids": eids_set,
        "session_num_neurons": session_num_neurons,
    }
    return train_dataset, val_dataset, test_dataset, meta_data

