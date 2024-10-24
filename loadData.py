import pandas as pd
import numpy as np

sensor_tags_default = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
counter_tag_default = "COUNTER"


def getRange(df: pd.DataFrame, counter_tag: str, sensor_tags: list[str], start: int, end: int):
    total = end-start

    mask = (df[counter_tag] >= start) & (df[counter_tag] < end)

    toReturn = df.loc[mask, sensor_tags]

    if len(toReturn) != total:
        return pd.DataFrame()

    return toReturn


def loadFromFile(file_name="data/test.csv", start=0,
                 sample_size=10, sensor_tags=sensor_tags_default, counter_tag = counter_tag_default):
    
    df = pd.read_csv(file_name, header= 0)

    print("preserving tags: ", sensor_tags, f"({len(sensor_tags)})")

    

    seconds = []
    second = 0
    skipped = []

    print("Starting to process data...")

    for i in range(start, len(df) + sample_size + 1, sample_size):
        second += 1
        currentStart = i
        currentEnd = i + sample_size # exclusive

        result = getRange(df, counter_tag, sensor_tags, currentStart, currentEnd)

        if not result.empty:
            seconds.append(result.to_numpy(dtype=np.float32).T)

        else:
            print(f"Incomplete second {second}. Skipping...")
            skipped.append(str(second))

    print("processing completed, converting to numpy...")

    completed = np.array(seconds, dtype=np.float32)

    print("Array is ready! Final shape:")
    print(completed.shape)

    print(f"For a total of {len(seconds)} chunks (probably seconds).")

    print(f"Skipped {len(skipped)}: not considered: {','.join(skipped)}")

    return completed