import re
import string
import torch
import numpy as np
from datasets import load_metric
from torch.nn import functional as F
from rich import print
from rich.console import Console
from rich.highlighter import Highlighter
from rich.table import Table
from rich.text import Text


class AmswerHighlighter(Highlighter):
    def __init__(self):
        self.start = 0
        self.end = -1
        super().__init__()

    def highlight(self, text):
        text.stylize(f"black on #90EE90", self.start, self.end)


class RefHighlighter(Highlighter):
    def highlight(self, text):
        text.stylize(f" ")


class QuestionHighlighter(Highlighter):
    def highlight(self, text):
        text.stylize(f"bold")


answer_hl = AmswerHighlighter()
ref_hl = RefHighlighter()
question_hl = QuestionHighlighter()
console = Console()


def normalize_text(text):
    text = text.lower()

    # Remove punctuations
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)

    # Remove articles
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    text = re.sub(regex, " ", text)

    # Remove extra white space
    text = " ".join(text.split())
    return text


class EvalUtility():
    """
    Each `SquadExample` object contains the character level offsets for each
    token in its input paragraph. We use them to get back the span of text
    corresponding to the tokens between our predicted start and end tokens.
    All ground-truth answers are also present in each `SquadExample` object.
    We calculate the percentage of data points where the span of text obtained
    from model predictions matches one of the ground-truth answers.
    """

    def __init__(self, x_eval, squad_examples, model):
        self.model = model
        self.squad_examples = squad_examples
        self.input_ids = x_eval['input_ids']
        self.token_type_ids = x_eval['token_type_ids']
        self.attention_mask = x_eval['attention_mask']

        # self.set_rich_print()

    def results(self, logs=None):
        metric = load_metric('squad')
        with torch.no_grad():
            outputs_eval = self.model(input_ids=self.input_ids,
                                      token_type_ids=self.token_type_ids,
                                      attention_mask=self.attention_mask
                                      )

        pred_start = F.softmax(outputs_eval.start_logits,
                               dim=-1).cpu().detach().numpy()
        pred_end = F.softmax(outputs_eval.end_logits,
                             dim=-1).cpu().detach().numpy()
        count = 0
        eval_examples_no_skip = [_ for _ in self.squad_examples
                                 if _.skip is False]
        for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
            squad_eg = eval_examples_no_skip[idx]
            offsets = squad_eg.context_token_to_char
            start = np.argmax(start)
            end = np.argmax(end) - 1
            if start >= len(offsets):
                continue

            pred_char_start = offsets[start][0]
            if end < len(offsets):
                pred_char_end = offsets[end][1]
                pred_ans = squad_eg.context[pred_char_start:pred_char_end]
                answer_hl.end = pred_char_end
            else:
                pred_ans = squad_eg.context[pred_char_start:]

            normalized_pred_ans = normalize_text(pred_ans)
            normalized_true_ans = [normalize_text(squad_eg.answer_text)]
            #                      for _ in squad_eg.all_answers]
            if normalized_pred_ans in normalized_true_ans:
                count += 1

            answer_hl.start = pred_char_start
            metric_value = metric.compute(
                predictions=[{'prediction_text': [normalized_pred_ans],
                              'id': 'xxx'}],
                references=[{'answers': {'answer_start': [pred_char_start],
                                         'text': normalized_true_ans},
                             'id': 'xxx'}]
            )
            console.rule(Text(f'{metric_value}'), style='magenta')
            print(':question:', question_hl(f'{squad_eg.question}'))
            print(':robot_face:', answer_hl(squad_eg.context))
            print(':white_check_mark:', ref_hl(f'{squad_eg.answer_text:30s}'))

    def set_rich_print(self):

        self.table = Table(title="Evaluation", show_lines=True)
        self.table.add_column("Question", justify="right", style="green")
        self.table.add_column("Model's answer", justify="right", style="cyan",
                              no_wrap=True)
        self.table.add_column("Reference", style="magenta")

    def show_table(self):
        console = Console()
        console.print(self.table)
