from multiprocessing import cpu_count


def get_num_workers(batch_size: int):
    num_workers = min(batch_size, 4, cpu_count())
    return num_workers


def round_to_divisor(x, divisor=16):
    return int(round(x / divisor) * divisor)
