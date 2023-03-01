import os
import subprocess
from collections import namedtuple
import json
import sys
import re
from loguru import logger

import torch
from transformers import BertTokenizer, BertForQuestionAnswering

from gpai import gpai
from model.bert import TtBertForQuestionAnswering
import models.bert.dle_squad as dle_squad
from models.utility_functions import set_FR

def get_squad_eval_examples():

    squad_dataset_path = f"{os.environ['BUDA_HOME']}/python_api_testing/models/bert/squad"
    assert os.path.isdir(squad_dataset_path), "You must first download the squad dataset! To do so, run squad_download.sh"

    squad_v1_dataset_path = f"{squad_dataset_path}/v1.1"
    assert "v1.1" in os.listdir(squad_dataset_path) and os.path.isdir(squad_v1_dataset_path), "Although the squad directory exists, 'v1.1' squad dataset should be within the squad directory. Potentially delete squad directory and re-download"

    # So far not requiring training json
    required_files = ["dev-v1.1.json", "evaluate-v1.1.py"]
    squad_v1_dataset_path_files = set(os.listdir(squad_v1_dataset_path))
    for f in required_files:
        assert f in squad_v1_dataset_path_files, "Although the squad v1.1 directory exists, it appears as if not all of the required files are there. Potentially delete it and re-download"

    eval_examples = dle_squad.read_squad_examples(input_file=f"{squad_v1_dataset_path}/dev-v1.1.json", is_training=False, version_2_with_negative=False)
    return eval_examples

def get_squad_eval_features_from_examples(eval_examples):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    eval_features = dle_squad.convert_examples_to_features(
        examples=eval_examples, tokenizer=tokenizer, max_seq_length=384, doc_stride=128, max_query_length=64, is_training=False
    )
    return eval_features


def prepare_squad_dataloader(input_ids, input_mask, segment_ids):
    class SquadDataset(torch.utils.data.Dataset):
        def __init__(self, input_ids, segment_ids, attention_mask):
            super().__init__()
            self.input_ids = input_ids
            self.segment_ids = segment_ids
            self.attention_mask = attention_mask

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):

            if self.attention_mask is not None:
                return (self.input_ids[idx], self.segment_ids[idx], self.attention_mask[idx])
            return self.input_ids[idx], self.segment_ids[idx]


    squad_batch = namedtuple("SquadBatch", "input_ids segment_ids attention_mask")
    def collate(inputs):
        size = len(inputs)
        return squad_batch(
            torch.cat(tuple(inputs[i][0].unsqueeze(0) for i in range(size))),
            torch.cat(tuple(inputs[i][1].unsqueeze(0) for i in range(size))),
            torch.cat(tuple(inputs[i][2].unsqueeze(0) for i in range(size))).unsqueeze(1).unsqueeze(1)
        )

    dataloader = torch.utils.data.DataLoader(
        SquadDataset(input_ids, input_mask, segment_ids),
        batch_size=32,
        collate_fn=collate
    )

    return dataloader


def squad_eval_inference(bert_name: str, run_evaluation: bool):
    assert "BUDA_HOME" in os.environ, "'BUDA_HOME' must be initialized"
    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained(bert_name, torchscript=False)
    hugging_face_reference_model.eval()
    model = TtBertForQuestionAnswering(2, 2, hugging_face_reference_model, device)

    eval_examples = get_squad_eval_examples()
    eval_features = get_squad_eval_features_from_examples(eval_examples)

    # Disaggregate eval_examples into tensors
    input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    example_index = torch.arange(input_ids.size(0), dtype=torch.long)

    # Create dataloader from features
    dataloader = prepare_squad_dataloader(input_ids, input_mask, segment_ids)

    # Run neural network inference
    start_logits_list = []
    end_logits_list = []
    for d in dataloader:
        input_ids, segment_ids, attention_mask = d
        out = gpai.tensor.untilize(model(input_ids, segment_ids))
        out = torch.Tensor(out.to(host).data()).reshape(out.shape())

        # Extract start and end logits from output
        tt_start_logits = out[..., :, 0].squeeze(1)
        tt_end_logits = out[..., :, 1].squeeze(1)

        start_logits_list.append(tt_start_logits)
        end_logits_list.append(tt_end_logits)

    # Convert the outputs into a form that can be passed into the squad evaluation script
    eval_results = []
    try:
        for i, example_index in enumerate(example_index):
            start_logits = start_logits_list[i].view(-1).detach().cpu().tolist()
            end_logits = end_logits_list[i].view(-1).detach().cpu().tolist()
            unique_id = eval_features[example_index].unique_id
            eval_results.append(dle_squad.RawResult(unique_id=unique_id, start_logits=start_logits, end_logits=end_logits))
    except IndexError:
        pass

    if run_evaluation:
        run_squad_eval(eval_examples, eval_features, eval_results)

    return eval_results

def run_squad_eval(squad_examples, squad_features, squad_results):
    args = dle_squad.DLESquadArgs(n_best_size=5, version_2_with_negative=False, do_lower_case=True, verbose_logging=False, max_answer_length=30)
    answers, nbest_answers = dle_squad.get_answers(squad_examples, squad_features, squad_results, args)

    squad_dataset_path = f"{os.environ['BUDA_HOME']}/python_api_testing/models/bert/squad/v1.1"
    results_folder = f"{squad_dataset_path}/results"

    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    output_prediction_file = os.path.join(results_folder, "predictions.json")
    output_nbest_file = os.path.join(results_folder, "nbest_predictions.json")
    with open(output_prediction_file, "w", encoding="utf-8") as file:
        file.write(json.dumps(answers, indent=4) + "\n")
    with open(output_nbest_file, "w", encoding="utf-8") as file:
        file.write(json.dumps(nbest_answers, indent=4) + "\n")

    # squad_v1_dataset_path = f"{squad_dataset_path}/v1.1"
    eval_out = subprocess.check_output([sys.executable, f"{squad_dataset_path}/evaluate-v1.1.py", f"{squad_dataset_path}/dev-v1.1.json", output_prediction_file])
    with open(f"{results_folder}/results.txt", "wb") as file:
        file.write(eval_out)

    eval_out = eval_out.decode()
    eval_out = re.findall(r"\d+\.\d+", eval_out)
    print(f"Exact match: {eval_out[0]}, F1 score: {eval_out[1]}")

if __name__ == "__main__":
    # Initialize the device
    device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, 0)
    gpai.device.InitializeDevice(device)
    host = gpai.device.GetHost()
    set_FR(0)
    results = squad_eval_inference("prajjwal1/bert-tiny", True)
    gpai.device.CloseDevice(device)
