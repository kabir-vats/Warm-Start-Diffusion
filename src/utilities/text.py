import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union

import h5py
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BertModel, BertTokenizer

from src.utilities.utils import get_logger


log = get_logger(__name__)


def get_llama_embedding(text, tokenizer=None, model=None, pooling=True, last_layer=True):
    """
    get the embedding of the text using the llama model and tokenizer
    text: str, the text to be embedded
    tokenizer: llama tokenizer
    model: llama model
    pooling: bool, whether to use mean pooling or take the last token's embedding
    last_layer: bool, whether to use the last layer or the middle layer
    """
    inputs = tokenizer(text, return_tensors="pt")

    with torch.inference_mode():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states

    if last_layer:
        target_hidden_state = hidden_states[-1]

    else:
        # take the middle one
        target_hidden_state = hidden_states[len(hidden_states) // 2]

    # Mask padding tokens
    attention_mask = inputs["attention_mask"]
    masked_embeddings = target_hidden_state * attention_mask.unsqueeze(-1)

    if pooling:
        # Compute mean pooling (ignoring padding)
        sentence_embedding = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

    else:
        # take the last token's embedding
        sentence_embedding = masked_embeddings[:, -1, :]

    return sentence_embedding[0].cpu().numpy()


def get_bert_embeddings(text, tokenizer, model, max_length=512):
    # Tokenize input text
    tokens = tokenizer.encode(text, add_special_tokens=True, max_length=max_length, truncation=True)
    # Convert tokens to tensor
    tokens_tensor = torch.tensor([tokens]).to(model.device)
    # Get BERT model output
    with torch.inference_mode():
        outputs = model(tokens_tensor)
        # Extract embeddings for [CLS] token (first token)
        embeddings = outputs[0][:, 0, :].squeeze().cpu().numpy()
    return embeddings


def get_dict_hash(d, length=8):
    """Create deterministic hash from a dictionary."""
    dhash = hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()
    return dhash[:length]


def get_or_create_embeddings(
    corpus, model_name, save_dir, history_length=0, force_recreate=False, metadata: Dict = None
):
    """
    Load embeddings if they exist, otherwise create and save them.

    Args:
        corpus: List of texts to embed
        model_name: Name of the model to use for embeddings (e.g. 'bert-base-uncased', "Meta-Llama-3.1-8B")
        save_dir: Directory to save embeddings
        history_length: Number of previous messages to include in the context
        force_recreate: Whether to recreate embeddings even if they exist
        metadata: Optional metadata to save with the embeddings (and verify when loading)

    Returns:
        List of numpy arrays containing embeddings
    """
    save_path = None
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_filename = str(model_name)
        if metadata is not None:
            save_filename += str(get_dict_hash(metadata))
        if history_length > 0:
            save_filename += f"-history{history_length}"
        save_filename += "-embeddings.h5"
        save_path = save_dir / save_filename
        save_path.parent.mkdir(parents=True, exist_ok=True)

    # Try to load existing embeddings
    if save_dir is not None and save_path.exists() and not force_recreate:
        log.info(f"Loading existing embeddings from {save_path}")
        try:
            with h5py.File(save_path, "r") as f:
                if metadata is not None:
                    for key, value in metadata.items():
                        value_f = None if f.attrs.get(key) == "None" else f.attrs.get(key)
                        if value_f != value:
                            log.warning(
                                f"Metadata mismatch: {key}={value_f} (type={type(value_f)}) vs {value} (type={type(value)})"
                            )
                            raise ValueError("Metadata mismatch")
                num_saved = f.attrs.get("num_embeddings", 0)
                if num_saved == len(corpus):  # Only use if complete
                    embeddings = f["embeddings"][:]  # Load embeddings into memory
                    log.info(f"Loaded {num_saved} embeddings (shape={embeddings.shape} from {save_path}.")
                    return embeddings  # [f['embeddings'][i] for i in range(num_saved)]
                else:
                    log.warning(f"Found incomplete embeddings ({num_saved} vs {len(corpus)} needed)")
        except Exception as e:
            log.warning(f"Error loading embeddings: {e}")

    # Create new embeddings
    log.info(f"Computing {model_name} embeddings for text data ({save_path=})...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_kwargs = dict()
    embed_kwargs = dict()
    embed_type_id = None
    model_name_base = model_name.rsplit("_")[0]  # split off any "_pool" or "_last" suffix
    if model_name_base != model_name:
        embed_type_id = model_name.rsplit("_")[-1]
    if "bert" in model_name.lower():
        if os.environ.get("PSCRATCH") is not None:
            maybe_from = os.path.join(os.environ["PSCRATCH"], "huggingface", model_name_base)
            if os.path.exists(maybe_from):
                # Useful on SLURM when the model is already downloaded and network is not available/slow
                model_name_base = maybe_from
                log.info(f"Loading BERT model locally from {model_name_base}")

        model_class = BertModel
        tokenizer_class = BertTokenizer
        embedding_func = get_bert_embeddings
        embed_kwargs["max_length"] = 512
    elif "llama" in model_name.lower():
        from huggingface_hub import login
        from huggingface_hub.hf_api import HfFolder

        token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
        if token is None:
            raise ValueError("Please set the HUGGINGFACE_TOKEN environment variable to use LLama.")
        if os.environ.get("HF_HOME") is None:
            os.environ["HF_HOME"] = "~/.cache/huggingface"
        # logout()
        login(token=token)
        HfFolder.save_token(token)

        cache_dir = os.path.join(save_dir, model_name_base)
        model_class = AutoModelForCausalLM
        tokenizer_class = AutoTokenizer
        load_kwargs["cache_dir"] = cache_dir
        embedding_func = get_llama_embedding
        if embed_type_id is None:
            embed_kwargs.update({"pooling": True, "last_layer": True})
        else:
            embed_kwargs.update({"pooling": "pool" in embed_type_id, "last_layer": "last" in embed_type_id})
            log.info(f"Using pooling={embed_kwargs['pooling']} and last_layer={embed_kwargs['last_layer']}")
    else:
        raise ValueError(f"Unknown model name {model_name}")

    model = model_class.from_pretrained(model_name_base, **load_kwargs)
    model.eval()
    model.to(device)
    tokenizer = tokenizer_class.from_pretrained(model_name_base, **load_kwargs)

    text_features = []
    # embeddings = corpus.apply(lambda x: get_bert_embeddings(x, tokenizer=tokenizer, model=model, device=device))
    history = []
    for x in tqdm(corpus, desc=f"{model_name} embeddings", total=len(corpus)):
        if history_length > 0:
            history.append(x)
            if len(history) > history_length:
                history.pop(0)
            # elif len(history) < history_length:
            #     text_features.append(None)  # Skip until we have enough history
            #     continue
            x = f"These are {len(history)} successive expert meteorologist forecast discussions:"
            for i, h in enumerate(history, 1):
                x += f"Forecast discussion {i}: {h}"
        embedding = embedding_func(x, tokenizer=tokenizer, model=model, **embed_kwargs)
        text_features.append(np.array(embedding, dtype=np.float32))

    embedding_dim = len(text_features[-1])
    try:
        # Save embeddings
        with h5py.File(save_path, "w") as f:
            embeddings_array = np.stack(text_features, axis=0)
            f.create_dataset(
                "embeddings",
                data=embeddings_array,
                compression="gzip",
                compression_opts=4,
                chunks=(min(1000, len(embeddings_array)), embedding_dim),
            )
            f.attrs["num_embeddings"] = len(embeddings_array)
            f.attrs["embedding_dim"] = embedding_dim
            if metadata is not None:
                for key, value in metadata.items():
                    f.attrs[key] = convert_to_saveable_type(value)

        log.info(f"Saved new embeddings to {save_path}")

    except Exception as e:
        log.error(f"Error saving embeddings: {e}")
        raise
    # del model
    return text_features


def convert_to_saveable_type(value: Any) -> Union[str, int, float, np.ndarray]:
    """Convert Python objects to HDF5-compatible types."""
    if isinstance(value, (str, int, float, np.ndarray)):
        return value
    elif isinstance(value, (datetime, np.datetime64)):
        return str(value)
    elif isinstance(value, (list, tuple)):
        return np.array(value)
    else:
        return str(value)
