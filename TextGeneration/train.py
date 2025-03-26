"""Code for calling the training of the model."""

from sys import argv

from TextGeneration.utils.files import json_to_schema, read_dir
from TextGeneration.utils.preprocessor import Preprocessor
from TextGeneration.utils.schemas import TrainingInputSchema


def main_train(file_str_path: str) -> None:
    """
    Call for training an n-gram language model.

    Do not modify its signature.
    You can modify the content.

    :param file_str_path: The path to the JSON that configures the training
    :return: None
    """
    # Reading input data
    training_schema = json_to_schema(
        file_str_path=file_str_path, input_schema=TrainingInputSchema
    )
    for training_line in read_dir(dir_path=training_schema.input_folder):
        _ = Preprocessor.clean(text=training_line)


if __name__ == "__main__":
    main_train(file_str_path=argv[1])
