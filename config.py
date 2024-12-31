from argparse import Namespace

config = Namespace(
    fn_data = 'iris_data.csv',
    max_epochs = 10,
    lr = 0.01,
    batch_size = 2,
    val_frac = 0.2,
    gpu_workers = 2,
    cpu_workers = 10
)