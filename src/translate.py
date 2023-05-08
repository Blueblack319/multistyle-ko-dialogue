import argparse
from deep_translator import GoogleTranslator
import glob
import json
import logging
import os
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from typing import List


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def check_similarity(
    originals: list[str], translateds: list[str], model, tokenizer
) -> int:
    similarity = []
    assert len(originals) == len(
        translateds
    ), "Original sentences and translated sentences are not matching!"

    sentences = [*originals, *translateds]

    encoded_input = tokenizer(
        sentences, padding=True, truncation=True, return_tensors="pt"
    )

    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])

    cos_sim = util.cos_sim(sentence_embeddings, sentence_embeddings)
    cos_sim = cos_sim.detach().cpu().numpy()

    for idx, row in enumerate(cos_sim):
        if idx < len(originals):
            similarity.append(row[idx + len(originals)])
    return similarity


def remove_noise(sentence: str) -> str:
    noise_words = ["키키", "하하"]
    for noise in noise_words:
        sentence = sentence.replace(noise, "")
    return sentence


def create_data_from_raw(path_src: str, path_dst: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained("jhgan/ko-sroberta-multitask")
    model = AutoModel.from_pretrained("jhgan/ko-sroberta-multitask")

    translator_ko_en = GoogleTranslator(source="ko", target="en")
    translator_en_ko = GoogleTranslator(source="en", target="ko")
    path = f"{path_src}/*"
    files = glob.glob(path)
    data = []

    # with open(files[0], mode="r") as f:
    #     originals = []
    #     for line in tqdm(f.readlines()):
    #         start_idx = line.find(":")
    #         if start_idx < 0:
    #             continue
    #         original = line[start_idx + 1 :].strip()
    #         original = remove_noise(original)
    #         originals.append(original)
    #     ko_en = translator_ko_en.translate_batch(originals)
    #     translateds = translator_en_ko.translate_batch(ko_en)
    #     cos_sim = check_similarity(originals, translateds, model, tokenizer)

    #     assert (
    #         len(cos_sim) == len(originals) == len(translateds)
    #     ), "Error: length is different!"

    #     for i in range(len(originals)):
    #         data.append(
    #             {
    #                 "original": originals[i],
    #                 "translated": translateds[i],
    #                 "similarity": float(cos_sim[i]),
    #             }
    #         )

    # if not os.path.exists(path_dst):
    #     os.makedirs(path_dst)

    # file_name = path_src[path_src.rfind("/") :]
    # file_path = path_dst + "/" + file_name + ".json"

    # with open(file_path, mode="w") as f:
    #     json.dump(data, f, ensure_ascii=False)

    for file in tqdm(files):
        with open(file, mode="r") as f:
            originals = []
            for line in f.readlines():
                start_idx = line.find(":")
                if start_idx < 0:
                    continue
                original = line[start_idx + 1 :].strip()
                original = remove_noise(original)
                originals.append(original)
            ko_en = translator_ko_en.translate_batch(originals)
            translateds = translator_en_ko.translate_batch(ko_en)
            cos_sim = check_similarity(originals, translateds, model, tokenizer)

            assert (
                len(cos_sim) == len(originals) == len(translateds)
            ), "Error: length is different!"

            for i in range(len(originals)):
                data.append(
                    {
                        "original": originals[i],
                        "translated": translateds[i],
                        "similarity": float(cos_sim[i]),
                    }
                )

    if not os.path.exists(path_dst):
        os.makedirs(path_dst)

    file_name = path_src[path_src.rfind("/") :]
    file_path = path_dst + "/" + file_name + ".json"

    with open(file_path, mode="w") as f:
        json.dump(data, f, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_src",
        type=str,
        required=True,
        default=None,
        help="The detail path of raw data folder",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        default="training",
        help="training | validation",
    )

    args = parser.parse_args()
    logger = logging.getLogger(__file__)

    dir_src = (
        "/home/intern/nas2/jhoon/multistyle-ko-dialogue/raw/subject_daily_dataset/data"
    )
    dir_dst = (
        "/home/intern/nas2/jhoon/multistyle-ko-dialogue/dataset/subject_daily_dataset"
    )
    # sources = [
    #     "kakao1",
    #     "kakao2",
    #     "kakao3",
    #     "kakao4",
    #     "band",
    #     "facebook",
    #     "instagram",
    #     "nateon",
    # ]
    path_src = dir_src + "/" + args.mode + "/" + args.path_src
    path_dst = dir_dst + "/" + args.mode

    create_data_from_raw(path_src, path_dst)
