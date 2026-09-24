import torch
from .base import BaseDataset, LengthStitchGroupedSampler, LengthGroupedSampler, SessionBatchSampler

def make_loader(dataset, 
                 batch_size, 
                 target = None,
                 pad_to_right = True,
                 sort_by_depth = False,
                 sort_by_region = False,
                 pad_value = 0.,
                 max_time_length = 5000,
                 max_space_length = 100,
                 bin_size = 0.05,
                 brain_region = 'all',
                 load_meta=False,
                 dataset_name = "ibl",
                 stitching = False,
                 shuffle = True,
                 seed = None):
    
    dataset = BaseDataset(dataset=dataset, 
                          target=target,
                          pad_value=pad_value,
                          max_time_length=max_time_length,
                          max_space_length=max_space_length,
                          bin_size=bin_size,
                          pad_to_right=pad_to_right,
                          dataset_name=dataset_name,
                          sort_by_depth = sort_by_depth,
                          sort_by_region = sort_by_region,
                          brain_region = brain_region,
                          load_meta=load_meta,
                          stitching=stitching
            )
    print(f"len(dataset): {len(dataset)}")

    # M2: a dataset holding more than one session gets session-homogeneous
    # batches (SessionBatchSampler). Anything with one EID -- every single-session
    # caller -- falls through to the original code path below, unchanged.
    eid_col = None
    raw = dataset.dataset
    if stitching and hasattr(raw, "column_names") and "eid" in raw.column_names:
        eid_col = [str(e).removesuffix("_aligned") for e in raw["eid"]]
    if eid_col is not None and len(set(eid_col)) > 1:
        if seed is None:
            raise ValueError("make_loader: a multi-session dataset needs an explicit seed "
                             "for SessionBatchSampler")
        batch_sampler = SessionBatchSampler(eid_col, batch_size=batch_size,
                                            shuffle=shuffle, seed=seed)
        print(f"SessionBatchSampler: {len(batch_sampler.session_order)} sessions, "
              f"{len(batch_sampler)} batches, shuffle={shuffle}")
        return torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler)
    if stitching:
        train_sampler = LengthStitchGroupedSampler(
            dataset=dataset, batch_size=batch_size, lengths=[sum(x["space_attn_mask"]) for x in dataset]
        )
    else:
        train_sampler = LengthGroupedSampler(
            dataset=dataset, batch_size=batch_size, 
            lengths=[sum(x["space_attn_mask"]) for x in dataset],
            shuffle=shuffle
        )

    # batch by neuron lengths makes multi-session training easier and NDT2 training faster
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=train_sampler, 
        batch_size=batch_size,
    )

    # the original data loader - each batch can contain different neuron lengths
    # dataloader = torch.utils.data.DataLoader(dataset,
    #                                             batch_size=batch_size,
    #                                             shuffle=shuffle)
    return dataloader
