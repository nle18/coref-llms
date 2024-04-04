from typing import List
from collections import defaultdict
from difflib import SequenceMatcher
from nltk import tokenize

import re
import copy
import difflib

from transformers import AutoTokenizer

from utils.io_utils import *


def get_dataset_readers(args):

    DATASET_READERS = {
        "doc_template": DocExampleDatasetReader,
        "qa_template": MentionExampleDatasetReader,
        "iterative_decoding_example": DocExampleIterativeDecodingDatasetReader,
    }
    gold_data = read_jsonl(args.eval_data)
    # TODO refactor with predicted mentions
    predicted_data = read_jsonl(args.eval_data)
    tokenizer = AutoTokenizer.from_pretrained("gpt2", add_prefix_space=True)
    dataset_reader = DATASET_READERS[args.prompt_template](
        gold_data, predicted_data, tokenizer
    )

    return dataset_reader


class DocExampleIterativeDecodingDatasetReader:
    """Dataset reader for doc-based examples, ITERATIVE DECODING VERSION. Inputs are either gold data
    (`gold_data`) and data output from mention detector (`predicted_data`).
    Functionalities of this class:
        - Mark the data based on types of template marker (e.g., XML, Markdown).
            -> This is implemented in the `split` function
        - Aggregate the predictions (i.e. generated text from LLMs) and output per-doc
        information (i.e., `predicted_clusters` and `gold_clusters`).
            -> This is implemented in the `aggregate` function


    ### Parameters
        gold_data (List[dict]): data with gold mentions and coref annotated
        predicted_data (List[dict]): data with predicted mentions
        tokenizer (AutoTokenizer)
        ID_format (str): Markdown, HTML, XLM tags
        ID_name (str): Either numerical (e.g. `cluster_1`) or longest string (e.g. `Hong_Kong_Disneyland`)
        max_example_len (int): Size of the example
    """

    def __init__(
        self,
        gold_data: List[dict],
        predicted_data: List[dict],
        tokenizer: AutoTokenizer,
        is_train: bool = False,
        ID_format: str = "markdown",
        ID_name: str = "numerical",
        max_example_len: int = 512,
    ):
        assert ID_format in ["xml", "markdown", "html"]
        assert ID_name in ["numerical", "longest_str"]
        self.gold_data = gold_data
        self.predicted_data = predicted_data
        self.split_data = None  # fill in later in .split
        self.tokenizer = tokenizer
        self.ID_format = ID_format
        self.ID_name = ID_name

    def _special_sort(self, ctx_mention_spans):
        ctx_mention_spans = sorted(ctx_mention_spans, key=lambda m: m[1])

        for i in range(len(ctx_mention_spans) - 1):

            if ctx_mention_spans[i][1] == ctx_mention_spans[i + 1][1]:

                for j in range(len(ctx_mention_spans) - 1):

                    if ctx_mention_spans[j][1] == ctx_mention_spans[j + 1][1]:

                        if ctx_mention_spans[j][0] < ctx_mention_spans[j + 1][0]:
                            ctx_mention_spans[j], ctx_mention_spans[j + 1] = (
                                ctx_mention_spans[j + 1],
                                ctx_mention_spans[j],
                            )

        return ctx_mention_spans

    def _is_nested(self, ctx_mention_spans):
        """Return a dict mapping span to nestedness"""

        # first sort by longest first
        ctx_mention_spans = sorted(ctx_mention_spans, key=lambda m: m[0])
        for i in range(len(ctx_mention_spans) - 1):
            if ctx_mention_spans[i][0] == ctx_mention_spans[i + 1][0]:
                for j in range(len(ctx_mention_spans) - 1):
                    if ctx_mention_spans[j][0] == ctx_mention_spans[j + 1][0]:
                        # swap
                        if ctx_mention_spans[j][1] < ctx_mention_spans[j + 1][1]:
                            ctx_mention_spans[j], ctx_mention_spans[j + 1] = (
                                ctx_mention_spans[j + 1],
                                ctx_mention_spans[j],
                            )
        # print("longest_first=", ctx_mention_spans)

        # then determine nestedness
        nestedness = defaultdict(dict)
        nestedness[str(ctx_mention_spans[0])] = {"nested_type": None}
        # this should always be the outside span
        cur_outside_span = ctx_mention_spans[0]
        for i in range(1, len(ctx_mention_spans)):
            cur_span = ctx_mention_spans[i]

            # if out of bound of cur_outside_span
            if cur_span[0] > cur_outside_span[1]:
                nestedness[str(cur_span)]["nested_type"] = None
                cur_outside_span = cur_span

            # if span is inside cur_outside_span
            elif cur_span[1] <= cur_outside_span[1]:
                nestedness[str(cur_outside_span)]["nested_type"] = "outside"
                if "nested_mentions" not in nestedness[str(cur_outside_span)]:
                    nestedness[str(cur_outside_span)]["nested_mentions"] = []
                nestedness[str(cur_outside_span)]["nested_mentions"].append(cur_span)
                nestedness[str(cur_span)] = {
                    "nested_type": "inside",
                    "nested_mentions": [str(cur_outside_span)],
                }

        print("nestedness=", nestedness)
        return nestedness

    def _generate_tags(self, mentionID, is_clusterID):

        if not is_clusterID:
            start_tag, end_tag = "[", "](#)"
        else:
            if self.ID_name == "numerical":
                start_tag, end_tag = "[", "](#cluster_{0})".format(mentionID)
            else:
                start_tag, end_tag = "[", "](#{0})".format(mentionID)

        return start_tag, end_tag

    def _get_context(
        self, token_IDs, clusters, mentions, mention_info, is_clusterID=True
    ):

        # get context
        ctx_tokens = self.tokenizer.convert_ids_to_tokens(token_IDs)

        # map mention to clusters; depending on self.ID_name, we have different clusterID
        mention_to_clusterID = dict()
        for i, cluster in enumerate(clusters):

            ID_name = ""
            if mention_info and self.ID_name == "longest_str":
                ID_name = ""
                for m in cluster:
                    m_str = mention_info[str(tuple(m))]["text"]
                    if len(m_str) > len(ID_name):
                        ID_name = m_str
                    ID_name = ID_name.replace(" ", "_")
            else:
                ID_name = i
            for m in cluster:
                mention_to_clusterID[str(m)] = ID_name

        # then annotate with cluster ID (and create mention_map along the way)
        mention_map = defaultdict(str)
        for mention in mentions:

            clusterID = mention_to_clusterID[str(mention)]
            start_tag, end_tag = self._generate_tags(clusterID, is_clusterID)
            ms, me = mention

            # special consideration for start
            ctx_tokens[ms] = " " + start_tag + ctx_tokens[ms][1:]

            # end tokens
            ctx_tokens[me] = ctx_tokens[me] + end_tag
            mention_list = self.tokenizer.convert_tokens_to_string(
                ctx_tokens[ms : me + 1]
            ).split(" ")
            mention_tokens = self.tokenizer.encode_plus(
                mention_list, is_split_into_words=True
            )["input_ids"]
            mention_str = self.tokenizer.decode(mention_tokens).strip()
            # mention_map[mention_str].append(mention)
            mention_map[str(mention)] = mention_str

        # convert to string, and back to tokens (re-tokenize)
        # print(ctx_tokens)
        context_list = self.tokenizer.convert_tokens_to_string(ctx_tokens).split(" ")
        context_tokens = self.tokenizer.encode_plus(
            context_list, is_split_into_words=True
        )["input_ids"]
        context_str = self.tokenizer.decode(context_tokens).strip()
        return context_tokens, context_str, mention_map

    def _get_output_priming(self, input_context_str: str) -> str:

        if self.ID_format == "markdown":
            first_hashtag = input_context_str.find("#")
            second_hashtag = input_context_str.find("#", first_hashtag + 1)
            output_priming = "{0}#cluster_0{1}".format(
                input_context_str[:first_hashtag],
                input_context_str[first_hashtag + 1 : second_hashtag + 1],
            )
            return output_priming
        else:
            raise NotImplementedError()

    def _remove_nesting(self, prompt, cur_span, nestedness, mention_map):
        nested_type = nestedness[cur_span]["nested_type"]
        if nested_type == "inside":

            # first get the outside mentions
            outside_m_str = mention_map[nestedness[cur_span]["nested_mentions"][0]]
            cur_m_str = mention_map[cur_span]

            # get the outside span up to the cur_m_str
            cur_m_str_idx = outside_m_str.find(cur_m_str)
            old_outside_m_str = outside_m_str[: cur_m_str_idx + len(cur_m_str)]

            # remove the oustide markers from oustide_m_str
            new_outside_m_str = outside_m_str[1 : -len("](#)")]
            cur_m_str_idx = new_outside_m_str.find(cur_m_str)
            new_outside_m_str = new_outside_m_str[: cur_m_str_idx + len(cur_m_str)]

            # then replace
            assert prompt[-len(old_outside_m_str) :] == old_outside_m_str
            prompt = prompt[: -len(old_outside_m_str)] + new_outside_m_str

        elif nested_type == "outside":

            # replace marked inside mentions with non-marked inside mentions
            cur_m_str = mention_map[cur_span]
            new_m_str = cur_m_str
            for inside_span in nestedness[cur_span]["nested_mentions"]:
                old_inside_m_str = mention_map[str(inside_span)]
                new_inside_m_str = old_inside_m_str[1 : -len("](#)")]
                new_m_str = new_m_str.replace(old_inside_m_str, new_inside_m_str)
            prompt = prompt[: -len(cur_m_str)] + new_m_str

        return prompt

    def split(self) -> List[dict]:
        split_data = []
        gold_i = 0
        for predicted_i, predicted_doc in enumerate(self.predicted_data):
            while predicted_doc["doc_key"] != self.gold_data[gold_i]["doc_key"]:
                gold_i += 1
            gold_doc = self.gold_data[gold_i]
            # for predicted_doc, gold_doc in zip(self.predicted_data, self.gold_data):
            assert predicted_doc["doc_key"] == gold_doc["doc_key"]

            # get predicted and gold mentions
            predicted_mentions, gold_mentions = [], []
            for mentions, clusters in (
                [predicted_mentions, predicted_doc["clusters"]],
                [gold_mentions, gold_doc["clusters"]],
            ):
                for cluster in clusters:
                    for m in cluster:
                        if m not in mentions:
                            mentions.append(m)
            predicted_mentions = self._special_sort(predicted_mentions)
            gold_mentions = self._special_sort(gold_mentions)

            # get nestedness info
            nestedness = self._is_nested(predicted_mentions)

            # sorted gold clusters
            gold_doc["clusters"] = sorted(gold_doc["clusters"], key=lambda c: c[0])

            # get contexts
            original_context_str = self.tokenizer.decode(gold_doc["sentences"]).strip()
            _, marked_context_str, mention_map = self._get_context(
                predicted_doc["sentences"],
                predicted_doc["clusters"],
                predicted_mentions,
                None,
                is_clusterID=False,
            )
            _, clusterIDs_context_str, _ = self._get_context(
                gold_doc["sentences"],
                gold_doc["clusters"],
                gold_mentions,
                None,  # gold_doc["mention_info"],
                is_clusterID=True,
            )

            # split the data according to mention (NOTE: this can be done with _get_context, but refactor later)
            prev_idx = 0
            doc_data = []
            # inside -> outside nested
            for i, (m_span, m_str) in enumerate(mention_map.items()):

                # first find mention in span
                m_idx = marked_context_str.find(m_str, prev_idx)
                output_priming = marked_context_str[: m_idx + len(m_str)]

                if nestedness[m_span]["nested_type"] != "inside":
                    prev_idx = m_idx + len(m_str)

                # remove nesting according to nesting types
                output_priming = self._remove_nesting(
                    output_priming, m_span, nestedness, mention_map
                )
                print(f"prompt for mention={m_str} is {output_priming}")

                doc_data.append(
                    {
                        "example_key": gold_doc["doc_key"] + f"_{i}",
                        "doc_key": gold_doc["doc_key"],
                        "mention": (m_str, m_span),
                        "original_context_str": original_context_str,
                        "input_context_str": marked_context_str,
                        "output_priming": output_priming,
                        "output_context_str": clusterIDs_context_str,
                        "predicted_mentions": predicted_mentions,
                        "gold_mentions": gold_mentions,
                        "gold_clusters": gold_doc["clusters"],
                        "mention_map": mention_map,
                        "nestedness": nestedness,
                        "doc_len": len(gold_doc["sentences"]),
                    }
                )
            split_data.append(doc_data)
        self.split_data = split_data
        return split_data

    def _get_positions(self, text):
        """Get the positions of entities in the text
        Eg. text="on [a cross-sea bridge connecting [[Hong Kong](#hk), [Zhuhai](#zhuhai),
        and [Macao](#macao)](#bridge)]"
        start_token=#, end_token=)
        Returns [(48,50), (62,68),...] (representation = [("hk", "zhuhai",...)])
        TODO: the "start_token", "end_token" business should be refactored
        """
        mentions = []
        text = text.strip()

        # first get the "start_token" and "end_token"
        start_token, end_token = (
            "\(#",  # "\(#" to avoid matching hashtags in the text itself
            ")",
        )

        # then find positions
        for m in re.finditer(start_token, text):

            start_idx = m.start()
            end_idx = text.find(end_token, start_idx)
            mentions.append((start_idx, end_idx))

        return mentions

    def _get_mentions(self, text):
        """Get the (mention, entityID pair) of entities in the text
        Eg. text="on [a cross-sea bridge connecting [[Hong Kong](#hk), [Zhuhai](#zhuhai),
        and [Macao](#macao)](#bridge)]"
        start_token=#
        end_token=)
        Returns [("[Hong Kong]", "hk"), ("[Zhuhai]", "zhuhai"),...]
        NOTE that this does not work entirely with nested mentions, but well enough
        for our purposes. Also this is a bit different from _get_positions. This
        whole parse_results business is convoluted, so most likely will need 2-3
        days of refactoring later
        """
        mentions = []
        text = text.strip()
        # the three variables below are specific to this prompt template
        identifier, start_m, end_id, align_idx = "#", "[", ")", 1
        for m in re.finditer(identifier, text):

            # get entity and mention
            start_id_idx = m.start()
            end_id_idx = text.find(end_id, start_id_idx)
            entity = text[start_id_idx + align_idx : end_id_idx]
            start_m_idx = text.rfind(start_m, 0, start_id_idx)
            mention = text[start_m_idx:start_id_idx]

            # we must replace all the entities in the mention with blank
            for prev_m, prev_e in mentions[::-1]:
                if (
                    "#{0}".format(prev_e) in mention
                    or "href={0}".format(prev_e) in mention
                ):
                    mention = mention.replace(prev_e, "")

            # finally we can add
            mentions.append((mention, entity))

        return mentions

    def _extract_clusters_from_unaligned_texts(
        self, generated_text, input_text, dev_example
    ):

        doc_key = dev_example["doc_key"]

        # first split generated_text and input_text into sentences
        if "\n" in generated_text:
            newline_idx = generated_text.find("\n")
            generated_text = generated_text[:newline_idx]
        generated_sents = tokenize.sent_tokenize(generated_text)
        input_sents = tokenize.sent_tokenize(input_text)
        assert len(generated_sents) == len(input_sents)
        try:
            input_mentions, generated_entities = [], []
            for generated_sent, input_sent in zip(generated_sents, input_sents):
                generated_sent = generated_sent.strip()
                input_sent = input_sent.strip()

                # here we can get the predictions more easily
                sent_generated_entities = self._get_mentions(generated_sent)
                sent_input_mentions = self._get_mentions(input_sent)

                new_sent_generated_entities = []
                e_counter = 0
                for (input_m, _) in sent_input_mentions:
                    if e_counter < len(sent_generated_entities):
                        generated_m, generated_e = sent_generated_entities[e_counter]
                    else:
                        generated_m, generated_e = "", ""
                    if input_m == generated_m:
                        new_sent_generated_entities.append((generated_m, generated_e))
                        e_counter += 1
                    else:
                        new_sent_generated_entities.append((input_m, ""))
                input_mentions += sent_input_mentions
                generated_entities += new_sent_generated_entities

            if (
                len(input_mentions)
                != len(dev_example["predicted_mentions"])
                != len(generated_entities)
            ):
                # print("Wrong position generation of doc_key={0}".format(doc_key))
                return None
            assert (
                len(input_mentions)
                == len(dev_example["predicted_mentions"])
                == len(generated_entities)
            )
            predicted_clusters_dict = defaultdict(list)
            for i, m in enumerate(dev_example["predicted_mentions"]):

                predicted_clusterID = generated_entities[i][1]
                if predicted_clusterID != "":
                    predicted_clusters_dict[predicted_clusterID].append(m)
            doc_predicted_clusters = [c for _, c in predicted_clusters_dict.items()]
            return doc_predicted_clusters
        except:
            return None

    def aggregate(self, generations: dict) -> List[dict]:
        """Aggregate generations into predicted clusters"""
        doc_data = []
        for example in self.split_data:

            try:
                doc_key = example["doc_key"]

                # for others
                if example["output_priming"][-8:] == "cluster_":
                    output_priming = example["output_priming"]
                # for constrained decoding
                else:
                    output_priming = example["output_priming"] + "clusters_"
                generated_text = "{0}{1}".format(
                    output_priming,
                    generations[doc_key]["generated_text"],
                ).strip()
                print(generated_text)
                # generated_text = generations[doc_key]["generated_text"].strip()
                input_text = example["input_context_str"].strip()
                generated_positions = self._get_positions(generated_text)
                input_positions = self._get_positions(input_text)
                assert len(input_positions) == len(example["predicted_mentions"])
                predicted_clusters = None
                # aligned: case where LM produces all mentions

                if (
                    len(generated_positions)
                    == len(input_positions)
                    == len(example["predicted_mentions"])
                ):
                    predicted_clusters_dict = defaultdict(list)
                    for i, m in enumerate(example["predicted_mentions"]):

                        e_start, e_end = generated_positions[i]
                        predicted_clusterID = generated_text[e_start:e_end]
                        predicted_clusters_dict[predicted_clusterID].append(m)
                    # predicted_clusters = [
                    #     c for _, c in predicted_clusters_dict.items() if len(c) > 1
                    # ]
                    predicted_clusters = [c for _, c in predicted_clusters_dict.items()]
                # misaligned: cases where LM does not produce all mentions
                else:
                    predicted_clusters = self._extract_clusters_from_unaligned_texts(
                        generated_text, input_text, example
                    )
                if predicted_clusters is not None:
                    doc_data.append(
                        {
                            "doc_key": doc_key,
                            "predicted_clusters": predicted_clusters,
                            "gold_clusters": example["gold_clusters"],
                        }
                    )
            except:
                print("Something wrong with doc_key={0}".format(doc_key))
                continue
        print(
            "Process {0} out of {1} docs ({2:0.2f}%)".format(
                len(doc_data),
                len(self.split_data),
                100 * len(doc_data) / len(self.split_data),
            )
        )
        return doc_data


class DocExampleDatasetReader:
    """Dataset reader for doc-based examples. Inputs are either gold data
    (`gold_data`) and data output from mention detector (`predicted_data`).
    Functionalities of this class:
        - Mark the data based on types of template marker (e.g., XML, Markdown).
            -> This is implemented in the `split` function
        - Aggregate the predictions (i.e. generated text from LLMs) and output per-doc
        information (i.e., `predicted_clusters` and `gold_clusters`).
            -> This is implemented in the `aggregate` function

    ### Parameters
        gold_data (List[dict]): data with gold mentions and coref annotated
        predicted_data (List[dict]): data with predicted mentions
        tokenizer (AutoTokenizer)
    """

    def __init__(
        self,
        gold_data: List[dict],
        predicted_data: List[dict],
        tokenizer: AutoTokenizer,
    ):
        self.gold_data = gold_data
        self.predicted_data = predicted_data
        self.split_data = None  # fill in later in .split
        self.tokenizer = tokenizer

    def _special_sort(self, ctx_mention_spans):
        ctx_mention_spans = sorted(ctx_mention_spans, key=lambda m: m[1])

        for i in range(len(ctx_mention_spans) - 1):

            if ctx_mention_spans[i][1] == ctx_mention_spans[i + 1][1]:

                for j in range(len(ctx_mention_spans) - 1):

                    if ctx_mention_spans[j][1] == ctx_mention_spans[j + 1][1]:

                        if ctx_mention_spans[j][0] < ctx_mention_spans[j + 1][0]:
                            ctx_mention_spans[j], ctx_mention_spans[j + 1] = (
                                ctx_mention_spans[j + 1],
                                ctx_mention_spans[j],
                            )

        return ctx_mention_spans

    def _get_context(
        self, token_IDs, clusters, mentions, mention_info, is_clusterID=True
    ):

        # get context
        ctx_tokens = self.tokenizer.convert_ids_to_tokens(token_IDs)

        # map mention to clusters; depending on self.ID_name, we have different clusterID
        mention_to_clusterID = dict()
        for i, cluster in enumerate(clusters):

            ID_name = i
            for m in cluster:
                mention_to_clusterID[str(m)] = ID_name

        # then annotate with cluster ID (and create mention_map along the way)
        mention_map = defaultdict(str)
        for mention in mentions:

            clusterID = mention_to_clusterID[str(mention)]
            start_tag = "["
            end_tag = f"](#cluster_{clusterID})" if is_clusterID else "](#)"
            ms, me = mention

            # special consideration for start
            if ctx_tokens[ms][0] == "Ġ":
                ctx_tokens[ms] = ctx_tokens[ms][0] + start_tag + ctx_tokens[ms][1:]
            else:
                ctx_tokens[ms] = " " + start_tag + ctx_tokens[ms][1:]

            # end tokens
            ctx_tokens[me] = ctx_tokens[me] + end_tag
            mention_list = self.tokenizer.convert_tokens_to_string(
                ctx_tokens[ms : me + 1]
            ).split(" ")
            mention_tokens = self.tokenizer.encode_plus(
                mention_list, is_split_into_words=True
            )["input_ids"]
            mention_str = self.tokenizer.decode(mention_tokens).strip()
            # mention_map[mention_str].append(mention)
            mention_map[str(mention)] = mention_str

        # convert to string, and back to tokens (re-tokenize)
        context_list = self.tokenizer.convert_tokens_to_string(ctx_tokens).split(" ")
        context_tokens = self.tokenizer.encode_plus(
            context_list, is_split_into_words=True
        )["input_ids"]
        context_str = self.tokenizer.decode(context_tokens).strip()
        return context_tokens, context_str, mention_map

    def _get_output_priming(self, input_context_str: str) -> str:

        first_hashtag = input_context_str.find("#")
        second_hashtag = input_context_str.find("#", first_hashtag + 1)
        output_priming = "{0}#cluster_0{1}".format(
            input_context_str[:first_hashtag],
            input_context_str[first_hashtag + 1 : second_hashtag + 1],
        )
        return output_priming

    def split(self) -> List[dict]:
        split_data = []
        gold_i = 0
        for predicted_i, predicted_doc in enumerate(self.predicted_data):
            while predicted_doc["doc_key"] != self.gold_data[gold_i]["doc_key"]:
                gold_i += 1
            gold_doc = self.gold_data[gold_i]
            # for predicted_doc, gold_doc in zip(self.predicted_data, self.gold_data):
            assert predicted_doc["doc_key"] == gold_doc["doc_key"]

            # get predicted and gold mentions
            predicted_mentions, gold_mentions = [], []
            for mentions, clusters in (
                [predicted_mentions, predicted_doc["clusters"]],
                [gold_mentions, gold_doc["clusters"]],
            ):
                for cluster in clusters:
                    for m in cluster:
                        if m not in mentions:
                            mentions.append(m)
            predicted_mentions = self._special_sort(predicted_mentions)
            gold_mentions = self._special_sort(gold_mentions)

            # sorted gold clusters
            gold_doc["clusters"] = sorted(gold_doc["clusters"], key=lambda c: c[0])

            # get contexts
            original_context_str = self.tokenizer.decode(gold_doc["sentences"]).strip()
            _, marked_context_str, mention_map = self._get_context(
                predicted_doc["sentences"],
                predicted_doc["clusters"],
                predicted_mentions,
                None,
                is_clusterID=False,
            )
            _, clusterIDs_context_str, _ = self._get_context(
                gold_doc["sentences"],
                gold_doc["clusters"],
                gold_mentions,
                None,  # gold_doc["mention_info"],
                is_clusterID=True,
            )

            split_data.append(
                {
                    "example_key": gold_doc["doc_key"],
                    "doc_key": gold_doc["doc_key"],
                    "original_context_str": original_context_str,
                    "input_context_str": marked_context_str,
                    "output_priming": self._get_output_priming(marked_context_str),
                    "output_context_str": clusterIDs_context_str,
                    "predicted_mentions": predicted_mentions,
                    "gold_mentions": gold_mentions,
                    "gold_clusters": gold_doc["clusters"],
                    "mention_map": mention_map,
                    "doc_len": len(gold_doc["sentences"]),
                }
            )
        self.split_data = split_data
        return split_data

    def _get_positions(self, text):
        """Get the positions of entities in the text
        Eg. text="on [a cross-sea bridge connecting [[Hong Kong](#hk), [Zhuhai](#zhuhai),
        and [Macao](#macao)](#bridge)]"
        start_token=#, end_token=)
        Returns [(48,50), (62,68),...] (representation = [("hk", "zhuhai",...)])
        """
        mentions = []
        text = text.strip()

        # first get the "start_token" and "end_token"
        start_token = "\(#"  # "\(#" to avoid matching hashtags in the text itself
        end_token = ")"

        # then find positions
        for m in re.finditer(start_token, text):

            start_idx = m.start()
            end_idx = text.find(end_token, start_idx)
            mentions.append((start_idx, end_idx))

        return mentions

    def _get_mentions(self, text):
        """Get the (mention, entityID pair) of entities in the text
        Eg. text="on [a cross-sea bridge connecting [[Hong Kong](#hk), [Zhuhai](#zhuhai),
        and [Macao](#macao)](#bridge)]"
        start_token=#
        end_token=)
        Returns [("[Hong Kong]", "hk"), ("[Zhuhai]", "zhuhai"),...]
        NOTE that this does not work entirely with nested mentions, but well enough
        for our purposes. Also this is a bit different from _get_positions. This
        whole parse_results business is convoluted, so most likely will need 2-3
        days of refactoring later
        """
        mentions = []
        text = text.strip()
        # the three variables below are specific to this prompt template
        identifier, start_m, end_id, align_idx = "#", "[", ")", 1
        for m in re.finditer(identifier, text):

            # get entity and mention
            start_id_idx = m.start()
            end_id_idx = text.find(end_id, start_id_idx)
            entity = text[start_id_idx + align_idx : end_id_idx]
            start_m_idx = text.rfind(start_m, 0, start_id_idx)
            mention = text[start_m_idx:start_id_idx]

            # we must replace all the entities in the mention with blank
            for prev_m, prev_e in mentions[::-1]:
                if f"#{prev_e}" in mention:
                    mention = mention.replace(prev_e, "")

            # finally we can add
            mentions.append((mention, entity))

        return mentions

    def _extract_clusters_from_unaligned_texts(
        self, generated_text, input_text, dev_example
    ):

        doc_key = dev_example["doc_key"]

        # first split generated_text and input_text into sentences
        if "\n" in generated_text:
            newline_idx = generated_text.find("\n")
            generated_text = generated_text[:newline_idx]
        generated_sents = tokenize.sent_tokenize(generated_text)
        input_sents = tokenize.sent_tokenize(input_text)
        # TODO: here we can align sentences according to Chao's suggestion:
        # (2) align sentences if necessary using Jaccard similarity
        # (3) resolve the hashtags locally
        try:
            assert len(generated_sents) == len(input_sents)
            input_mentions, generated_entities = [], []
            for generated_sent, input_sent in zip(generated_sents, input_sents):
                generated_sent = generated_sent.strip()
                input_sent = input_sent.strip()

                # here we can get the predictions more easily
                sent_generated_entities = self._get_mentions(generated_sent)
                sent_input_mentions = self._get_mentions(input_sent)
                new_sent_generated_entities = []
                e_counter = 0
                for (input_m, _) in sent_input_mentions:
                    if e_counter < len(sent_generated_entities):
                        generated_m, generated_e = sent_generated_entities[e_counter]
                    else:
                        generated_m, generated_e = "", ""
                    if input_m == generated_m:
                        new_sent_generated_entities.append((generated_m, generated_e))
                        e_counter += 1
                    else:
                        new_sent_generated_entities.append((input_m, ""))
                input_mentions += sent_input_mentions
                generated_entities += new_sent_generated_entities
            if (
                len(input_mentions)
                != len(dev_example["predicted_mentions"])
                != len(generated_entities)
            ):
                return None
            assert (
                len(input_mentions)
                == len(dev_example["predicted_mentions"])
                == len(generated_entities)
            )
            predicted_clusters_dict = defaultdict(list)
            for i, m in enumerate(dev_example["predicted_mentions"]):

                predicted_clusterID = generated_entities[i][1]
                if predicted_clusterID != "":
                    predicted_clusters_dict[predicted_clusterID].append(m)
            doc_predicted_clusters = [c for _, c in predicted_clusters_dict.items()]
            return doc_predicted_clusters
        except:
            return None

    def aggregate(self, generations: dict) -> List[dict]:
        """Aggregate generations into predicted clusters"""
        doc_data = []
        for example in self.split_data:

            # try:
            doc_key = example["doc_key"]
            generated_text = "{0}{1}".format(
                example["output_priming"],
                generations[doc_key]["generated_text"],
            ).strip()
            # generated_text = generations[doc_key]["generated_text"].strip()
            input_text = example["input_context_str"].strip()
            generated_positions = self._get_positions(generated_text)
            input_positions = self._get_positions(input_text)
            assert len(input_positions) == len(example["predicted_mentions"])
            predicted_clusters = None
            # aligned: case where LM produces all mentions

            if (
                len(generated_positions)
                == len(input_positions)
                == len(example["predicted_mentions"])
            ):
                predicted_clusters_dict = defaultdict(list)
                for i, m in enumerate(example["predicted_mentions"]):

                    e_start, e_end = generated_positions[i]
                    predicted_clusterID = generated_text[e_start:e_end]
                    predicted_clusters_dict[predicted_clusterID].append(m)

                predicted_clusters = [c for _, c in predicted_clusters_dict.items()]
            # misaligned: cases where LM does not produce all mentions
            else:
                predicted_clusters = self._extract_clusters_from_unaligned_texts(
                    generated_text, input_text, example
                )
            if predicted_clusters is not None:
                doc_data.append(
                    {
                        "doc_key": doc_key,
                        "predicted_clusters": predicted_clusters,
                        "gold_clusters": example["gold_clusters"],
                    }
                )
        # except:
        #     print("Something wrong with doc_key={0}".format(doc_key))
        #     continue
        print(
            "Process {0} out of {1} docs ({2:0.2f}%)".format(
                len(doc_data),
                len(self.split_data),
                100 * len(doc_data) / len(self.split_data),
            )
        )
        return doc_data


class MentionExampleDatasetReader:
    """Dataset reader for mention-based examples. Inputs are gold data (gold_data) '
    and data output from mention detector (predicted_data). Example template for
    this data class is something like "What does [mention a] refer to?", and the
    answer is a span [mention b].
    Main functionalities:
    - Mark the data based on types of template marker (e.g., XML, brackets, etc).
        -> This is mainly done in the `split` function
    - Aggregate the predictions (i.e. generated text from LLMs) and output per-doc
    information (i.e., predicted_clusters and gold_clusters).
        -> This is mainly done in the `aggregate` function

    ### Parameters
    gold_data: type List[dict]
    predicted_data: type List[dict]
    tokenizer: type AutoTokenizer
    is_train: type bool
        Whether storing training data or not. The main difference between
        train vs. test is train example will include gold antecedents related field,
        whereas test example may contain predicted mentions (which do not have
        gold antecedents)
    use_gold_mentions: type bool
        Whether to use gold mentions in evaluation data
    mention_marker: type str
        Whether to mark the mentions in the text. Options: None, brackets, xml
    max_example_len: type int
        Max context len for an individual example
    """

    def __init__(
        self,
        gold_data: List[dict],
        predicted_data: List[dict],
        tokenizer: AutoTokenizer,
        is_train: bool = False,
        use_gold_mentions: bool = False,
        mention_marker: str = None,
        max_example_len: int = 512,
    ):
        assert mention_marker in [None, "xml", "brackets"]
        self.gold_data = gold_data
        self.predicted_data = predicted_data
        self.is_train = is_train
        self.use_gold_mentions = use_gold_mentions
        self.split_data = None  # fill in later in .split
        self.tokenizer = tokenizer
        self.mention_marker = mention_marker
        self.max_example_len = max_example_len
        self.null_token = "N/A"
        self.null_indicies = [-1, -1]

    def _special_sort(self, ctx_mention_spans):
        ctx_mention_spans = sorted(ctx_mention_spans, key=lambda m: m[1])

        for i in range(len(ctx_mention_spans) - 1):

            if ctx_mention_spans[i][1] == ctx_mention_spans[i + 1][1]:

                for j in range(len(ctx_mention_spans) - 1):

                    if ctx_mention_spans[j][1] == ctx_mention_spans[j + 1][1]:

                        if ctx_mention_spans[j][0] < ctx_mention_spans[j + 1][0]:
                            ctx_mention_spans[j], ctx_mention_spans[j + 1] = (
                                ctx_mention_spans[j + 1],
                                ctx_mention_spans[j],
                            )

        return ctx_mention_spans

    def _generate_tags(self, is_anaphor=False):

        if is_anaphor:
            return "*", "*"

        if self.mention_marker == "brackets":
            start_tag, end_tag = "[", "]"
        else:
            start_tag, end_tag = "", ""

        return start_tag, end_tag

    def _get_context(self, anaphor, token_IDs, mentions, sentence_end_idx):
        """### Parameters
        anaphor: list of span indicies
        token_IDs: list of IDs
        sentence_end_idx: index of where the sentence ends

        Returns context_str
        """

        anaphor_end = anaphor[1]
        ctx_end = anaphor_end + 1
        for sent_end in sentence_end_idx:
            if sent_end > anaphor_end:
                ctx_end = sent_end + 1
                break
        ctx_start = max(0, ctx_end - self.max_example_len)

        # finally get context
        ctx_tokens = self.tokenizer.convert_ids_to_tokens(token_IDs[ctx_start:ctx_end])

        # find all mentions within this context
        ctx_mention_spans = []
        for [start, end] in mentions:
            if ctx_start <= start and end < ctx_end:
                ctx_mention_spans.append([start, end])

        # sort according to end (so []_0 is the closest to anaphor). This makes
        # engineering more convenient (ie nested mentions)
        ctx_mention_spans = self._special_sort(ctx_mention_spans)

        # then annotate with unique identifiers (and create mention_map along the way)
        mention_map = defaultdict(list)
        for i, mention in enumerate(ctx_mention_spans):

            is_anaphor = mention == anaphor

            # modify anaphor_str
            start_tag, end_tag = self._generate_tags(is_anaphor)
            ms, me = mention[0] - ctx_start, mention[1] - ctx_start

            # special consideration for start
            if ctx_tokens[ms][0] == "Ġ":
                ctx_tokens[ms] = ctx_tokens[ms][0] + start_tag + ctx_tokens[ms][1:]
            else:
                ctx_tokens[ms] = start_tag + ctx_tokens[ms]

            # end tokens
            ctx_tokens[me] = ctx_tokens[me] + end_tag

            # add to mention map
            mention_list = self.tokenizer.convert_tokens_to_string(
                ctx_tokens[ms : me + 1]
            ).split(" ")
            mention_tokens = self.tokenizer.encode_plus(
                mention_list, is_split_into_words=True
            )["input_ids"]
            mention_str = self.tokenizer.decode(mention_tokens).strip()
            mention_map[mention_str].append(mention)

        # convert to string, and back to tokens (re-tokenize)
        context_list = self.tokenizer.convert_tokens_to_string(ctx_tokens).split(" ")
        context_tokens = self.tokenizer.encode_plus(
            context_list, is_split_into_words=True
        )["input_ids"]
        context_str = self.tokenizer.decode(context_tokens).strip()

        return context_str, mention_map

    def _map_anaphor_to_antecedents(self, doc):

        mentions = doc["mentions"]
        clusters = doc["clusters"]
        anaphor_map = defaultdict(list)
        for anaphor in mentions:
            for cluster in clusters:
                if anaphor in cluster:

                    mention_idx = cluster.index(anaphor)
                    anaphor_map[str(anaphor)] = cluster[:mention_idx]

        return anaphor_map

    def _get_gold_antecedents(self, anaphor, anaphor_str, anaphor_map, token_IDs):

        if len(anaphor_map[str(anaphor)]) == 0:
            gold_antecedents = [anaphor_str]
        else:
            gold_antecedents = []
            for ante in anaphor_map[str(anaphor)]:
                if ante == self.null_indicies:
                    gold_antecedents.append(anaphor_str)
                else:
                    ante_str = self.tokenizer.decode(
                        token_IDs[ante[0] : ante[1] + 1]
                    ).strip()
                    # if self.mention_marker == "brackets":
                    #     ante_str = "[{0}]".format(ante_str)
                    gold_antecedents.append(ante_str)

        return gold_antecedents

    def _split_train(self) -> list:
        split_data = []
        for doc in self.gold_data:

            # first get doc['mentions'] and doc['sentence_end_indexes']
            mentions = []
            for c in doc["clusters"]:
                for m in c:
                    if m not in mentions:
                        mentions.append(m)
            doc["mentions"] = mentions
            sent_end_idx = []
            for i, sent_idx in enumerate(doc["sentence_map"]):
                if i < len(doc["sentence_map"]) - 1:
                    if doc["sentence_map"][i + 1] != sent_idx:
                        sent_end_idx.append(i)
            doc["sentence_end_indexes"] = sent_end_idx

            # first we want to map each anaphor to (all) its antecedents
            anaphor_map = self._map_anaphor_to_antecedents(doc)

            # create an example for each mention / anaphor
            for i, anaphor in enumerate(doc["mentions"]):

                # first get anaphor
                anaphor_tokens = doc["sentences"][anaphor[0] : anaphor[1] + 1]
                anaphor_str = "*{0}*".format(
                    self.tokenizer.decode(anaphor_tokens).strip()
                )

                # then get context
                context_str, _ = self._get_context(
                    anaphor,
                    doc["sentences"],
                    doc["mentions"],
                    doc["sentence_end_indexes"],
                )

                # get gold antecedents
                gold_antecedents = self._get_gold_antecedents(
                    anaphor, anaphor_str, anaphor_map, doc["sentences"]
                )

                split_data.append(
                    {
                        "example_key": doc["doc_key"] + "_" + str(i),
                        "doc_key": doc["doc_key"],
                        "context_str": context_str,
                        "anaphor_str": anaphor_str,
                        "gold_antecedents_str": gold_antecedents,
                        # "mention_info": doc["mention_info"],
                    }
                )
        self.split_data = split_data
        return split_data

    def _split_test(self) -> List[dict]:
        """Split the document into window of size max_example_len. The context will
        surround the anaphor.
        """
        split_data = []
        for gold_doc, predicted_doc in zip(self.gold_data, self.predicted_data):

            # first get predicted mentions
            predicted_mentions = []
            for cluster in predicted_doc["clusters"]:
                for m in cluster:
                    if m not in predicted_mentions:
                        predicted_mentions.append(m)
            predicted_mentions = self._special_sort(predicted_mentions)

            # get predicted_mention_info (for easier time mapping str -> index in aggregate)
            predicted_mention_info = defaultdict(list)
            for m_index in predicted_mentions:
                m_tokens = predicted_doc["sentences"][m_index[0] : m_index[1] + 1]
                m_str = self.tokenizer.decode(m_tokens).strip()
                predicted_mention_info[m_str].append(m_index)

            # also create gold_mention_info
            gold_mention_info = defaultdict(list)
            for _, m in gold_doc["mention_info"].items():
                gold_mention_info[m["text"]].append(m["index"])

            if self.use_gold_mentions:
                # first we want to map each anaphor to (all) its antecedents
                anaphor_map = self._map_anaphor_to_antecedents(gold_doc)

            # create an example for each (predicted) mention (anaphor)
            for i, anaphor in enumerate(predicted_mentions):

                # first get anaphor
                anaphor_tokens = predicted_doc["sentences"][anaphor[0] : anaphor[1] + 1]
                anaphor_str = anaphor_str = "*{0}*".format(
                    self.tokenizer.decode(anaphor_tokens).strip()
                )

                # then get context
                context_str, _ = self._get_context(
                    anaphor,
                    predicted_doc["sentences"],
                    predicted_mentions,
                    gold_doc["sentence_end_indexes"],
                )

                data = {
                    "example_key": predicted_doc["doc_key"] + "_" + str(i),
                    "doc_key": predicted_doc["doc_key"],
                    "context_str": context_str,
                    "anaphor_str": anaphor_str,
                    "anaphor_indicies": anaphor,
                    "predicted_mention_info": predicted_mention_info,
                    "gold_mention_info": gold_mention_info,
                }

                if self.use_gold_mentions:
                    # get gold antecedents
                    data["gold_antecedents_str"] = self._get_gold_antecedents(
                        anaphor, anaphor_str, anaphor_map, gold_doc["sentences"]
                    )
                # add to list
                split_data.append(data)

        self.split_data = split_data
        return split_data

    def split(self) -> List[dict]:
        """Split the document into window of size max_example_len. The context will
        surround the anaphor.
        """
        if self.is_train:
            return self._split_train()
        else:
            return self._split_test()

    def _merge_clusters(self, clusters: list) -> list:

        merged_clusters = []
        while len(merged_clusters) != len(clusters):

            if len(merged_clusters) > 0:
                clusters = copy.deepcopy(merged_clusters)

            for cluster in clusters:

                # initialize clusters if empty
                yes_cluster = False
                for merged_cluster in merged_clusters:

                    # first idenfity if cluster and merged_cluster are the same
                    for span in cluster:
                        if span in merged_cluster:
                            yes_cluster = True
                            break

                    if yes_cluster:
                        for span in cluster:
                            if span not in merged_cluster:
                                merged_cluster.append(span)
                        break

                if not yes_cluster:
                    merged_clusters.append([span for span in cluster])

        # for each cluster, make sure spans are unique
        # we also sort each cluster; this step is optional
        for i in range(len(merged_clusters)):
            unique_cluster = []
            for span in merged_clusters[i]:
                if span not in unique_cluster:
                    unique_cluster.append(span)
            merged_clusters[i] = sorted(unique_cluster, key=lambda s: s[0])
        merged_clusters = sorted(merged_clusters, key=lambda c: c[0])

        return merged_clusters

    def _extract_mIndex_from_mStr(self, generated_text, mention_info, anaphor):
        if generated_text in mention_info:

            # try to get the closest mentions to the anaphor
            candidate_mentions = mention_info[generated_text]
            for i, m in enumerate(candidate_mentions):

                # case 1: essentially cataphora
                if i == 0 and m[0] > anaphor[1]:
                    return m

                # case 2: in-between
                elif i < len(candidate_mentions) - 1:
                    next_m = candidate_mentions[i + 1]
                    if m[1] < anaphor[0] < next_m[0]:
                        return m

                # case 3: at the very end
                else:
                    return m
        return None

    def aggregate(self, generations: dict) -> List[dict]:
        """Aggregate the predictions, example-level, into document-level example.
        Note that this is only available for non-train dataset
        """
        doc_data = []
        for gold_doc in self.gold_data:

            # first get all the generations in the doc
            output_examples = [
                v for k, v in generations.items() if gold_doc["doc_key"] in k
            ]
            input_examples = [
                e for e in self.split_data if e["doc_key"] == gold_doc["doc_key"]
            ]

            # map generated text to index and add to mention_pairs, if available
            # NOTE: technically, extract from "gold_mention_info" is cheating, because
            # we are using gold information. But, this can be (easily?) replaced by
            # some span extraction algorithm
            mention_pairs = []
            for input_e, output_e in zip(input_examples, output_examples):
                # TODO: this is where the "resolver" should be
                generated_text = output_e["generated_text"]
                if generated_text[0] == "[" and generated_text[-1] == "]":
                    generated_text = generated_text[1:-1]
                anaphor = input_e["anaphor_indicies"]

                m = self._extract_mIndex_from_mStr(
                    generated_text, input_e["gold_mention_info"], anaphor
                )

                if m is None:
                    m = self._extract_mIndex_from_mStr(
                        generated_text, input_e["predicted_mention_info"], anaphor
                    )

                if m is not None:
                    mention_pairs.append([anaphor, m])

            predicted_clusters = self._merge_clusters(mention_pairs)

            doc_data.append(
                {
                    "doc_key": gold_doc["doc_key"],
                    "predicted_clusters": predicted_clusters,
                    "gold_clusters": gold_doc["clusters"],
                }
            )
        return doc_data
