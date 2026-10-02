"""Joint schema head on a decoder backbone, for the Von nano-decoder variant.

The head (``EvidenceRoutingLayer``, ``JointSchemaHead``) and the record packing
(``encode_record``) are adapted from Cloudflare's Clef release
(https://huggingface.co/Cloudflare/clef-flash, ``joint_schema_model.py``,
Apache-2.0). Media handling is dropped: Von is text-only. Everything else is kept
so that a head trained here scores records the same way Clef does: one logit per
allowed option, softmax per question, with a lexical prior from the backbone's
output embeddings gated against an evidence-routed residual.

Two entry points produce the same packed layout:

* ``encode_record``  -- a Jev-wire request body (``state`` + ``questions``).
* ``encode_row``     -- one Von training row ``{state, question, options, label[, target]}``,
                        converted to a one-question record. Question type is inferred
                        from the option ids (yes/no -> noul, 0..K-1 -> score, else choice)
                        so the head's type embedding sees all three types in training.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as functional

SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)
QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}
NOUL_DEFAULT_CRITERIA = {
    "true": "The proposition is true or the answer is yes.",
    "false": "The proposition is false or the answer is no.",
}
_YES = {"yes", "true"}
_NO = {"no", "false"}


def render(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def question_options(question: Dict[str, Any]) -> List[Tuple[str, Any]]:
    question_type = str(question["type"])
    if question_type == "noul":
        criteria = dict(NOUL_DEFAULT_CRITERIA)
        criteria.update(question.get("criteria") or {})
        return [(key, criteria[key]) for key in ("true", "false")]
    if question_type == "choice":
        return sorted((str(key), value) for key, value in question["criteria"].items())
    return [(str(index), value) for index, value in enumerate(question["criteria"])]


@dataclass(frozen=True)
class EncodedQuestion:
    question_id: str
    question_type: int
    question_span: Tuple[int, int]
    option_spans: Tuple[Tuple[int, int], ...]
    option_ids: Tuple[str, ...]


@dataclass(frozen=True)
class EncodedRecord:
    input_ids: Tuple[int, ...]
    questions: Tuple[EncodedQuestion, ...]
    record_id: str


def _tokens(tokenizer: Any, text: str) -> List[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def encode_record(
    tokenizer: Any,
    record: Dict[str, Any],
    max_length: int = 8192,
    max_state_tokens: Optional[int] = None,
) -> EncodedRecord:
    """Pack ``{state, questions:{id: {type, instructions, criteria}}}`` into one sequence.

    Layout: system prompt, ``STATE:`` + state, ``SCHEMA FIELDS`` with one block per
    question (instruction span, then one span per option), then a closed empty
    ``<think>`` and ``JOINT SCHEMA DECISIONS:``. The state is the only part that is
    truncated to fit ``max_length``; a schema that does not fit on its own is an error.
    """
    schema_ids = _tokens(tokenizer, "\n\nSCHEMA FIELDS:\n")
    questions: List[EncodedQuestion] = []
    for question_index, (question_id, question) in enumerate(record["questions"].items()):
        schema_ids.extend(_tokens(
            tokenizer, f"\nFIELD {question_index + 1}\nID: {question_id}\nTYPE: {question['type']}\nINSTRUCTION: "))
        question_start = len(schema_ids)
        instructions = question.get("instructions")
        if instructions is None or instructions == "":
            instructions = str(question_id)
        schema_ids.extend(_tokens(tokenizer, render(instructions)))
        question_end = len(schema_ids)
        schema_ids.extend(_tokens(tokenizer, "\nALLOWED OPTIONS:\n"))
        option_spans: List[Tuple[int, int]] = []
        option_ids: List[str] = []
        for option_index, (option_id, description) in enumerate(question_options(question)):
            schema_ids.extend(_tokens(tokenizer, f"OPTION {option_index + 1}: "))
            option_start = len(schema_ids)
            semantics: Dict[str, Any] = {"option_id": option_id}
            if description is not None:
                semantics["description"] = description
            schema_ids.extend(_tokens(tokenizer, render(semantics)))
            option_spans.append((option_start, len(schema_ids)))
            option_ids.append(option_id)
            schema_ids.extend(_tokens(tokenizer, "\n"))
        schema_ids.extend(_tokens(tokenizer, "END FIELD\n"))
        questions.append(EncodedQuestion(
            question_id=str(question_id),
            question_type=QUESTION_TYPES[str(question["type"])],
            question_span=(question_start, question_end),
            option_spans=tuple(option_spans),
            option_ids=tuple(option_ids),
        ))

    prefix_ids = _tokens(tokenizer, f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n")
    suffix_ids = _tokens(tokenizer, "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:")
    state_ids = _tokens(tokenizer, render(record["state"]))
    if max_state_tokens is not None:
        state_ids = state_ids[:max_state_tokens]
    fixed_length = len(prefix_ids) + len(schema_ids) + len(suffix_ids)
    if fixed_length > max_length:
        raise ValueError(f"schema requires {fixed_length} tokens before state; maximum is {max_length}")
    state_ids = state_ids[: max_length - fixed_length]
    offset = len(prefix_ids) + len(state_ids)
    shifted = tuple(
        EncodedQuestion(
            question_id=q.question_id,
            question_type=q.question_type,
            question_span=(q.question_span[0] + offset, q.question_span[1] + offset),
            option_spans=tuple((s + offset, e + offset) for s, e in q.option_spans),
            option_ids=q.option_ids,
        )
        for q in questions
    )
    input_ids = tuple(prefix_ids + state_ids + schema_ids + suffix_ids)
    if not input_ids or not shifted:
        raise ValueError("record produced no model input or questions")
    return EncodedRecord(input_ids=input_ids, questions=shifted, record_id=str(record.get("id", "unknown")))


def infer_row_type(option_ids: Sequence[str]) -> str:
    """yes/no or true/false pairs are noul; 0..K-1 ids are score; everything else is choice."""
    lowered = [str(o).strip().lower() for o in option_ids]
    if len(lowered) == 2 and set(lowered) in ({"yes", "no"}, {"true", "false"}):
        return "noul"
    if len(lowered) >= 2 and all(o.isdigit() for o in lowered) and sorted(int(o) for o in lowered) == list(range(len(lowered))):
        return "score"
    return "choice"


def row_to_record(row: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Von training row -> (one-question Jev-wire record, target option id as the head sees it)."""
    opts = row["options"]
    ids = [str(o["id"]) for o in opts]
    qtype = infer_row_type(ids)
    label = str(row["label"])
    question: Dict[str, Any] = {"type": qtype, "instructions": row.get("question") or ""}
    if qtype == "noul":
        criteria = {}
        for o in opts:
            key = "true" if str(o["id"]).strip().lower() in _YES else "false"
            criteria[key] = o["description"]
        question["criteria"] = criteria
        target = "true" if label.strip().lower() in _YES else "false"
    elif qtype == "score":
        order = sorted(opts, key=lambda o: int(str(o["id"])))
        question["criteria"] = [o["description"] for o in order]
        target = str(int(label))
    else:
        question["criteria"] = {str(o["id"]): o["description"] for o in opts}
        target = label
    return {"id": row.get("id", "row"), "state": row["state"], "questions": {"q": question}}, target


def soft_target_for(row: Dict[str, Any], option_ids: Sequence[str]) -> Optional[List[float]]:
    """Reorder a row's ``target`` distribution (given in the row's option order) to the packed option order."""
    soft = row.get("target")
    opts = row["options"]
    if not (isinstance(soft, list) and len(soft) == len(opts)):
        return None
    total = float(sum(soft))
    if total <= 0:
        return None
    qtype = infer_row_type([str(o["id"]) for o in opts])
    by_id: Dict[str, float] = {}
    for o, p in zip(opts, soft):
        oid = str(o["id"])
        if qtype == "noul":
            oid = "true" if oid.strip().lower() in _YES else "false"
        elif qtype == "score":
            oid = str(int(oid))
        by_id[oid] = float(p) / total
    return [by_id.get(oid, 0.0) for oid in option_ids]


def collate_records(records: List[EncodedRecord], pad_token_id: int, device: torch.device) -> Dict[str, Any]:
    maximum = max(len(r.input_ids) for r in records)
    input_ids = torch.full((len(records), maximum), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(records), maximum), dtype=torch.long, device=device)
    for i, r in enumerate(records):
        n = len(r.input_ids)
        input_ids[i, :n] = torch.tensor(r.input_ids, device=device)
        attention_mask[i, :n] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "records": records}


class EvidenceRoutingLayer(torch.nn.Module):
    """Option queries cross-attend over the packed sequence (pre-norm, residual, GELU FFN)."""

    def __init__(self, width: int, heads: int, feedforward: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.query_norm = torch.nn.LayerNorm(width)
        self.memory_norm = torch.nn.LayerNorm(width)
        self.attention = torch.nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.attention_dropout = torch.nn.Dropout(dropout)
        self.feedforward_norm = torch.nn.LayerNorm(width)
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(width, feedforward), torch.nn.GELU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(feedforward, width), torch.nn.Dropout(dropout))

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        normalized_memory = self.memory_norm(memory)
        routed, _ = self.attention(self.query_norm(queries), normalized_memory, normalized_memory, need_weights=False)
        queries = queries + self.attention_dropout(routed)
        return queries + self.feedforward(self.feedforward_norm(queries))


class JointSchemaHead(torch.nn.Module):
    """logit = lexical_prior + sigmoid(gate) * (scale * cos(field, option) + MLP([f, o, f*o, |f-o|]))."""

    def __init__(self, hidden_size: int, width: int, routing_layers: int, layers: int, heads: int,
                 feedforward: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.config = {"hidden_size": hidden_size, "width": width, "routing_layers": routing_layers,
                       "layers": layers, "heads": heads, "feedforward": feedforward}
        self.hidden_norm = torch.nn.LayerNorm(hidden_size)
        self.memory_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.question_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.option_question_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.global_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.option_context_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.option_lexical_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.type_embedding = torch.nn.Embedding(3, width)
        self.evidence_layers = torch.nn.ModuleList(
            [EvidenceRoutingLayer(width, heads, feedforward, dropout) for _ in range(routing_layers)])
        self.option_summary_norm = torch.nn.LayerNorm(width)
        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerDecoderLayer(d_model=width, nhead=heads, dim_feedforward=feedforward, dropout=dropout,
                                             activation="gelu", batch_first=True, norm_first=True)
            for _ in range(layers)])
        self.field_norm = torch.nn.LayerNorm(width)
        self.option_norm = torch.nn.LayerNorm(width)
        self.residual_scorer = torch.nn.Sequential(
            torch.nn.Linear(width * 4, width), torch.nn.GELU(), torch.nn.Dropout(dropout), torch.nn.Linear(width, 1))
        self.prior_logit_scale = torch.nn.Parameter(torch.zeros(()))
        self.joint_logit_scale = torch.nn.Parameter(torch.zeros(()))
        self.residual_gate = torch.nn.Parameter(torch.zeros(()))

    @staticmethod
    def _mean_span(values: torch.Tensor, span: Tuple[int, int]) -> torch.Tensor:
        start, end = span
        return values[start:end].mean(dim=0)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                records: List[EncodedRecord], output_embedding_weight: torch.Tensor) -> List[List[torch.Tensor]]:
        results: List[List[torch.Tensor]] = []
        normalized_hidden = self.hidden_norm(hidden_states)
        for b, record in enumerate(records):
            n = int(attention_mask[b].sum().item())
            seq = normalized_hidden[b, :n]
            memory = self.memory_projection(seq).unsqueeze(0)
            global_vector = seq[-1]
            question_vectors = torch.stack([self._mean_span(seq, q.question_span) for q in record.questions])
            type_ids = torch.tensor([q.question_type for q in record.questions], device=hidden_states.device)

            option_contexts: List[torch.Tensor] = []
            lexical_options: List[torch.Tensor] = []
            option_counts: List[int] = []
            for q in record.questions:
                option_contexts.append(torch.stack([self._mean_span(seq, s) for s in q.option_spans]))
                lexical_options.append(torch.stack(
                    [output_embedding_weight[input_ids[b, s:e]].mean(dim=0) for s, e in q.option_spans]))
                option_counts.append(len(q.option_spans))

            option_queries = [
                self.option_context_projection(ctx) + self.option_lexical_projection(lex)
                + self.option_question_projection(question_vectors[i]).unsqueeze(0)
                for i, (ctx, lex) in enumerate(zip(option_contexts, lexical_options))]
            routed = torch.cat(option_queries, dim=0).unsqueeze(0)
            for layer in self.evidence_layers:
                routed = layer(routed, memory)
            split_options = list(torch.split(routed[0], option_counts, dim=0))

            base_fields = self.question_projection(question_vectors)
            summaries = []
            for field, options in zip(base_fields, split_options):
                w = torch.softmax(torch.matmul(options, field) / math.sqrt(options.shape[-1]), dim=0)
                summaries.append(torch.sum(w.unsqueeze(-1) * options, dim=0))
            fields = (base_fields + self.option_summary_norm(torch.stack(summaries))
                      + self.global_projection(global_vector).unsqueeze(0) + self.type_embedding(type_ids)).unsqueeze(0)
            for layer in self.layers:
                fields = layer(fields, memory)
            fields = self.field_norm(fields[0])

            record_logits: List[torch.Tensor] = []
            for i, (field, lexical, routed_options) in enumerate(zip(fields, lexical_options, split_options)):
                anchor = functional.normalize(question_vectors[i] + global_vector, dim=-1)
                prior = self.prior_logit_scale.clamp(max=math.log(100.0)).exp() * torch.matmul(
                    functional.normalize(lexical, dim=-1), anchor)
                options = self.option_norm(routed_options)
                rep = field.unsqueeze(0).expand_as(options)
                cosine = functional.cosine_similarity(rep, options, dim=-1)
                residual = self.residual_scorer(torch.cat([rep, options, rep * options, torch.abs(rep - options)], dim=-1)).squeeze(-1)
                joint = self.joint_logit_scale.clamp(max=math.log(100.0)).exp() * cosine + residual
                record_logits.append(prior + torch.sigmoid(self.residual_gate) * joint)
            results.append(record_logits)
        return results


def text_model_of(language_model: Any) -> Any:
    """The text trunk of a Qwen3.5 backbone, through a PEFT wrapper if present."""
    base = language_model.get_base_model() if hasattr(language_model, "get_base_model") else language_model
    text = base.model
    if hasattr(text, "language_model"):
        text = text.language_model
    return base, text


class DecoderHeadModel(torch.nn.Module):
    """Backbone (optionally LoRA-wrapped) + JointSchemaHead; forward returns per-record, per-question logits."""

    def __init__(self, language_model: Any, head: JointSchemaHead) -> None:
        super().__init__()
        self.language_model = language_model
        self.head = head

    def forward(self, batch: Dict[str, Any]) -> List[List[torch.Tensor]]:
        base, text = text_model_of(self.language_model)
        out = text(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False, return_dict=True)
        return self.head(out.last_hidden_state.float(), batch["input_ids"], batch["attention_mask"], batch["records"],
                         base.get_output_embeddings().weight.float())


def systemone_answer(question: Dict[str, Any], probabilities: Dict[str, float]) -> Dict[str, Any]:
    """Per-option probabilities for one question -> SystemOne answer body."""
    if question["type"] == "noul":
        return {"type": "noul", "noul": round(probabilities["true"], 4)}
    if question["type"] == "choice":
        options = [str(o) for o in question["criteria"]]
        pick = max(options, key=probabilities.__getitem__)
        return {"type": "choice", "choice": pick, "confidence": round(probabilities[pick], 4),
                "probabilities": {o: round(probabilities[o], 4) for o in options}}
    levels = [str(i) for i in range(len(question["criteria"]))]
    return {"type": "score",
            "score": round(sum(i * probabilities[lvl] for i, lvl in enumerate(levels)), 4),
            "confidence": round(max(probabilities[lvl] for lvl in levels), 4),
            "legend": dict(zip(levels, question["criteria"])),
            "probabilities": {lvl: round(probabilities[lvl], 4) for lvl in levels}}
