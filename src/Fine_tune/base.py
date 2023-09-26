# First, import all necessary modules and libraries
from abc import ABC, abstractmethod
from enum import Enum
import torch
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Optional, Union
import pandas as pd
from transformers.tokenization_utils_base import BatchEncoding
from datasets import Dataset
import pandas as pd
from functools import partial
from typing import Optional, cast
import logging



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Assuming err module and logger are also defined elsewhere
# import err
# import logger

# Define your first class
class BaseEmbeddingGenerator(ABC):
    def __init__(self, use_case: Enum, model_name: str, batch_size: int = 100, **kwargs):
        self.__use_case = self._parse_use_case(use_case=use_case)
        self.__model_name = model_name
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == torch.device("cpu"):
            logger.warning(
                "No available GPU has been detected. The use of GPU acceleration is "
                "strongly recommended. You can check for GPU availability by running "
                "`torch.cuda.is_available()`"
            )
        self.__batch_size = batch_size
        logger.info(f"Downloading pre-trained model '{self.model_name}'")
        try:
            # Extract the base model name and the number
            base_model_name = ''.join([i for i in self.model_name if not i.isdigit()])
            number = ''.join([i for i in self.model_name if i.isdigit()])

            if base_model_name == "fine_tuned_model_circor":
                self.__model = AutoModel.from_pretrained(f"/data/chenxi/llm-feature-engeneering/src/Fine_tune/circor/model/{number}/checkpoint-960", **kwargs).to(self.device)
            elif base_model_name == "fine_tuned_model_heart_disease":
                self.__model = AutoModel.from_pretrained(f"/data/chenxi/llm-feature-engeneering/src/Fine_tune/heart_disease/model/{number}/checkpoint-300", **kwargs).to(self.device)
            elif base_model_name == "fine_tuned_model_diabetes":
                self.__model = AutoModel.from_pretrained(f"/data/chenxi/llm-feature-engeneering/src/Fine_tune/diabetes/model/{number}/checkpoint-780", **kwargs).to(self.device)
            elif base_model_name == "fine_tuned_model_euca":
                self.__model = AutoModel.from_pretrained(f"/data/chenxi/llm-feature-engeneering/src/Fine_tune/euca/model/{number}/checkpoint-720", **kwargs).to(self.device)
            else:
                self.__model = AutoModel.from_pretrained(self.model_name, **kwargs).to(self.device)

        except OSError:
            raise ValueError(f"Unable to load model from {self.model_name}")
        except Exception as e:
            raise e

    @property
    def use_case(self) -> str:
        return self.__use_case

    @property
    def model_name(self) -> str:
        return self.__model_name

    @property
    def model(self):
        return self.__model

    @property
    def device(self) -> torch.device:
        return self.__device

    @property
    def batch_size(self) -> int:
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, new_batch_size: int) -> None:
        err_message = "New batch size should be an integer greater than 0."
        if not isinstance(new_batch_size, int):
            raise TypeError(err_message)
        elif new_batch_size <= 0:
            raise ValueError(err_message)
        else:
            self.__batch_size = new_batch_size
            logger.info(f"Batch size has been set to {new_batch_size}.")

    @staticmethod
    def _parse_use_case(use_case: Union[Enum, str]) -> str:
        if isinstance(use_case, Enum):
            uc_area = use_case.__class__.__name__.split("UseCases")[0]
            uc_task = use_case.name
        elif isinstance(use_case, str):
            uc_area, uc_task = use_case.split('.')
        else:
            raise TypeError("use_case should be either of type Enum or str.")
        return f"{uc_area}.{uc_task}"


    def _get_embedding_vector(
        self, batch: Dict[str, torch.Tensor], method
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            outputs = self.model(**batch)
        # (batch_size, seq_length/or/num_tokens, hidden_size)
        if method == "cls_token":  # Select CLS token vector
            embeddings = outputs.last_hidden_state[:, 0, :]
        elif method == "avg_token":  # Select avg token vector
            embeddings = torch.mean(outputs.last_hidden_state, 1)
        else:
            raise ValueError(f"Invalid method = {method}")
        return {"embedding_vector": embeddings.cpu().numpy().astype(float)}

    @staticmethod
    def check_invalid_index(field: Union[pd.Series, pd.DataFrame]) -> None:
        if (field.index != field.reset_index(drop=True).index).any():
            if isinstance(field, pd.DataFrame):
                raise err.InvalidIndexError("DataFrame")
            else:
                raise err.InvalidIndexError(str(field.name))

    @abstractmethod
    def __repr__(self) -> str:
        pass

# Define your second class
class NLPEmbeddingGenerator(BaseEmbeddingGenerator):
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  use_case={self.use_case},\n"
            f"  model_name='{self.model_name}',\n"
            f"  tokenizer_max_length={self.tokenizer_max_length},\n"
            f"  tokenizer={self.tokenizer.__class__},\n"
            f"  model={self.model.__class__},\n"
            f"  batch_size={self.batch_size},\n"
            f")"
        )

    def __init__(self, use_case: Enum, model_name: str, tokenizer_max_length: int = 512, **kwargs):
        super(NLPEmbeddingGenerator, self).__init__(
            use_case=use_case, model_name=model_name, **kwargs
        )
        self.__tokenizer_max_length = tokenizer_max_length
        # We don't check for the tokenizer's existence since it is coupled with the corresponding model
        # We check the model's existence in `BaseEmbeddingGenerator`
        logger.info(f"Downloading tokenizer for '{self.model_name}'")
        if self.model_name == "fine_tuned_model":
             self.__tokenizer = AutoTokenizer.from_pretrained(
                'distilbert-base-uncased', model_max_length=self.tokenizer_max_length
            )
        else:
            self.__tokenizer = AutoTokenizer.from_pretrained(
                'distilbert-base-uncased', model_max_length=self.tokenizer_max_length
            )

    @property
    def tokenizer(self):
        return self.__tokenizer

    @property
    def tokenizer_max_length(self) -> int:
        return self.__tokenizer_max_length

    def tokenize(self, batch: Dict[str, List[str]], text_feat_name: str) -> BatchEncoding:
        return self.tokenizer(
            batch[text_feat_name],
            padding=True,
            truncation=True,
            max_length=self.tokenizer_max_length,
            return_tensors="pt",
        ).to(self.device)


# Define your third class
class EmbeddingGeneratorForNLPSequenceClassification(NLPEmbeddingGenerator):
    def __init__(self, use_case: Enum, model_name: str, tokenizer_max_length: int = 512, **kwargs):
        super(EmbeddingGeneratorForNLPSequenceClassification, self).__init__(
            use_case=use_case,
            model_name=model_name,
            tokenizer_max_length=tokenizer_max_length,
            **kwargs
        )
    @classmethod
    def from_use_case(cls, model_name: str, tokenizer_max_length: int, use_case: str = None):
        if use_case == "NLP.SequenceClassification":
            return cls(use_case=use_case, model_name=model_name, tokenizer_max_length=tokenizer_max_length)
        else:
            raise ValueError(f"Unknown use case: {use_case}")


    def generate_embeddings(
        self,
        text_col: pd.Series,
        class_label_col: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Obtain embedding vectors from your text data using pre-trained large language models.

        :param text_col: a pandas Series containing the different pieces of text.
        :param class_label_col: if this column is passed, the sentence "The classification label
        is <class_label>" will be appended to the text in the `text_col`.
        :return: a pandas Series containing the embedding vectors.
        """
        if not isinstance(text_col, pd.Series):
            raise TypeError("text_col must be a pandas Series")

        self.check_invalid_index(field=text_col)

        if class_label_col is not None:
            if not isinstance(class_label_col, pd.Series):
                raise TypeError("class_label_col must be a pandas Series")
            df = pd.concat({"text": text_col, "class_label": class_label_col}, axis=1)
            prepared_text_col = df.apply(
                lambda row: f" The classification label is {row['class_label']}. {row['text']}",
                axis=1,
            )
            ds = Dataset.from_dict({"text": prepared_text_col})
        else:
            ds = Dataset.from_dict({"text": text_col})

        ds.set_transform(partial(self.tokenize, text_feat_name="text"))
        logger.info("Generating embedding vectors")
        ds = ds.map(
            lambda batch: self._get_embedding_vector(batch, "cls_token"),
            batched=True,
            batch_size=self.batch_size,
        )
        return cast(pd.DataFrame, ds.to_pandas())["embedding_vector"]
