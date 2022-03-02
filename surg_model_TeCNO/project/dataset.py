import csv
from functools import partial
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.layers.experimental.preprocessing import Resizing

AUTOTUNE = tf.data.experimental.AUTOTUNE

@tf.function
def preprocess_input_fn(data, preprocessor):

    img_as_float = tf.cast(data["image"], tf.float32)
    preprocessed_img = preprocessor(img_as_float)

    new_data = {key:preprocessed_img if key=="image" else val for key,val in data.items()}

    return new_data

@tf.function
def read_image(data):

    img = tf.io.read_file(data["image_path"])
    img = tf.image.decode_image(img, channels=3, dtype=tf.uint8)
    img.set_shape((None, None, 3))

    new_data = {"image": img}
    new_data.update(data)

    return new_data

@tf.function
def resize_map(data):

    rescaled_img = Resizing(224, 224)(data["image"])

    new_data = {key:rescaled_img if key=="image" else val for key,val in data.items()}

    return new_data


def backbone_dataset(data_root,
                     batch_size):
    
    data_root = Path(data_root)

    csv_files = list((data_root / "data_csv").glob("*"))
    csv_files.sort()
    
    datasets = list()
    csv_file_names = list()

    for csv_file in csv_files:
        frames = list()
        labels = list()
        frame_ids = list()

        with open(csv_file) as f:
            reader = csv.reader(f)
            for frame_id, row in enumerate(reader):
                if frame_id == 0:
                    continue
                frames.append(row[0])
                labels.append(int(row[1]))
                frame_ids.append(frame_id)

        frames = list(map(lambda path: str(data_root / path), frames))
        
        csv_file_names.append(csv_file.name)
        datasets.append(tf.data.Dataset.from_tensor_slices((frames, labels, frame_ids))
                        .map(lambda img, label, frame_id: {"image_path":img, "label":label, "frame_id":frame_id})
                        .map(read_image, num_parallel_calls=AUTOTUNE)
                        .batch(batch_size)
                        .map(resize_map, num_parallel_calls=AUTOTUNE)
                        .map(partial(preprocess_input_fn, preprocessor=preprocess_input), num_parallel_calls=AUTOTUNE)
                        .prefetch(AUTOTUNE)
        )
    
    return csv_file_names, datasets