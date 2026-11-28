"""Bind prompt changes to an exact source snapshot without executing it."""

from __future__ import annotations

import ast
import copy
import io
import json
import re
import tokenize
from dataclasses import asdict, dataclass
from typing import Any

from app.prompt_optimizer.models import digest

_TEXT_NAME = re.compile(r"(?:^|_)(?:prompts?|instructions?|template|description|system_message|developer_message)(?:_|$)", re.I)
_SLOTS = re.compile(r"\$?\{\{.*?\}\}|\$?\{[^{}]*\}|%(?:\([^)]+\))?[#0 +\-]*\d*(?:\.\d+)?[diouxXeEfFgGcrsa%]", re.S)
_SECRETS = re.compile(r"(?:github_pat_[A-Za-z0-9_]{10,}|gh[pousr]_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,})")


def _check_source_context(agent_code: str, start: int, end: int) -> None:
    line_start = agent_code.rfind("\n", 0, start) + 1
    line_end = agent_code.find("\n", end)
    line_end = len(agent_code) if line_end < 0 else line_end
    if agent_code[line_start:start].strip() or agent_code[end:line_end].strip():
        raise ValueError("Patch fragment must cover complete source lines")
    lines = agent_code.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    def offset(position):
        row, column = position
        return offsets[min(row - 1, len(lines))] + column

    try:
        f_strings = []
        for token in tokenize.generate_tokens(io.StringIO(agent_code).readline):
            if token.type == tokenize.FSTRING_START:
                f_strings.append(offset(token.start))
            elif token.type == tokenize.FSTRING_END:
                first = f_strings.pop()
                if first <= start < offset(token.end):
                    raise ValueError("Patch target is inside an f-string, not an agent statement")
            if token.type in (tokenize.STRING, tokenize.COMMENT):
                first, last = offset(token.start), offset(token.end)
                if first <= start < last:
                    raise ValueError("Patch target is inside a string or comment, not an agent statement")
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        raise ValueError("Agent snapshot could not be checked for Python string boundaries") from exc


class _PromptMasker(ast.NodeTransformer):
    """Mask recognized text parameters while retaining executable expressions."""

    def __init__(self, source: str):
        self.lines = source.splitlines(keepends=True)
        self.spans: set[tuple[int, int]] = set()

    def offset(self, line: int, column: int) -> int:
        return sum(map(len, self.lines[:line - 1])) + len(
            self.lines[line - 1].encode("utf-8")[:column].decode("utf-8")
        )

    def mask_text(self, node: ast.AST) -> ast.AST:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("format", "format_map"):
            node.func.value = self.mask_text(node.func.value)
            return node
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str) or isinstance(node, ast.JoinedStr)):
            return node
        span = (self.offset(node.lineno, node.col_offset), self.offset(node.end_lineno, node.end_col_offset))
        if span in self.spans:
            return node
        self.spans.add(span)
        if isinstance(node, ast.Constant):
            node.value = "PROMPT_TEXT:" + json.dumps(sorted(_SLOTS.findall(node.value)), ensure_ascii=False)
        else:
            literal = "".join(part.value for part in node.values if isinstance(part, ast.Constant))
            expressions = [part for part in node.values if isinstance(part, ast.FormattedValue)]
            node.values = [ast.Constant("PROMPT_TEXT:" + json.dumps(sorted(_SLOTS.findall(literal)), ensure_ascii=False)), *expressions]
        return node

    def visit_Assign(self, node: ast.Assign):
        if any(_TEXT_NAME.search(getattr(target, "id", getattr(target, "attr", ""))) for target in node.targets):
            node.value = self.mask_text(node.value)
        return self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        if node.value is not None and _TEXT_NAME.search(getattr(node.target, "id", getattr(node.target, "attr", ""))):
            node.value = self.mask_text(node.value)
        return self.generic_visit(node)

    def visit_keyword(self, node: ast.keyword):
        if node.arg and _TEXT_NAME.search(node.arg):
            node.value = self.mask_text(node.value)
        return self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict):
        role = next((value.value for key, value in zip(node.keys, node.values)
                     if isinstance(key, ast.Constant) and key.value == "role"
                     and isinstance(value, ast.Constant)), None)
        for index, key in enumerate(node.keys):
            if isinstance(key, ast.Constant) and isinstance(key.value, str) and (
                _TEXT_NAME.search(key.value)
                or key.value == "content" and role in ("system", "developer", "user")
            ):
                node.values[index] = self.mask_text(node.values[index])
        return self.generic_visit(node)

    def visit_Tuple(self, node: ast.Tuple):
        if len(node.elts) == 2 and isinstance(node.elts[0], ast.Constant) and node.elts[0].value in ("system", "developer", "user", "human"):
            node.elts[1] = self.mask_text(node.elts[1])
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        name = getattr(node.func, "id", getattr(node.func, "attr", ""))
        if node.args and (name == "from_template" or name.lower().endswith("prompttemplate")):
            node.args[0] = self.mask_text(node.args[0])
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        decorators = [item.func if isinstance(item, ast.Call) else item for item in node.decorator_list]
        if any(getattr(item, "id", getattr(item, "attr", "")) in ("tool", "function_tool") for item in decorators):
            if node.body and isinstance(node.body[0], ast.Expr):
                node.body[0].value = self.mask_text(node.body[0].value)
        return self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_JoinedStr(self, node: ast.JoinedStr):
        # Expressions inside f-strings are executable code and stay unchanged.
        return node


def _normalized_fragment(fragment: str) -> tuple[str, str, str]:
    source = fragment
    try:
        tree = ast.parse(source)
    except IndentationError:
        source = "if True:\n" + fragment
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise ValueError("Patch fragments must be complete Python statements") from exc
    except SyntaxError as exc:
        raise ValueError("Patch fragments must be complete Python statements") from exc
    original = ast.dump(tree, include_attributes=False)
    masker = _PromptMasker(source)
    masked = masker.visit(copy.deepcopy(tree))
    if not masker.spans:
        raise ValueError("No supported prompt literal found in patch fragment")
    skeleton = source
    for start, end in sorted(masker.spans, reverse=True):
        skeleton = skeleton[:start] + "PROMPT_LITERAL" + skeleton[end:]
    return ast.dump(masked, include_attributes=False), skeleton, original


@dataclass(frozen=True)
class SourcePromptPatch:
    source_hash: str
    start: int
    end: int
    before: str
    after: str
    rationale: str
    report_index: int
    hypothesis_id: str
    evidence_refs: tuple[dict[str, Any], ...]

    @classmethod
    def create(
        cls, agent_code: str, *, before: str, after: str, rationale: str,
        report_index: int, hypothesis_id: str, evidence_refs: list[dict[str, Any]],
        max_growth_chars: int = 1500,
    ) -> SourcePromptPatch:
        if not all(isinstance(value, str) and value.strip() for value in (before, after, rationale)):
            raise ValueError("Patch requires non-empty before, after and rationale")
        start = agent_code.find(before)
        if start < 0 or agent_code.find(before, start + 1) >= 0:
            raise ValueError("Patch before fragment must occur exactly once in agent_code")
        _check_source_context(agent_code, start, start + len(before))
        if len(after) - len(before) > max_growth_chars:
            raise ValueError("Prompt growth limit exceeded")
        old_tree, old_skeleton, old_actual = _normalized_fragment(before)
        new_tree, new_skeleton, new_actual = _normalized_fragment(after)
        if old_tree != new_tree or old_skeleton != new_skeleton:
            raise ValueError("Patch may change only prompt text; code and template variables must remain unchanged")
        if old_actual == new_actual:
            raise ValueError("Patch must make a non-empty prompt change")
        if _SECRETS.search(after):
            raise ValueError("Potential secret embedded in candidate prompt")
        return cls(digest(agent_code), start, start + len(before), before, after, rationale,
                   report_index, hypothesis_id, tuple(copy.deepcopy(evidence_refs)))

    def apply(self, agent_code: str) -> str:
        if digest(agent_code) != self.source_hash or agent_code[self.start:self.end] != self.before:
            raise ValueError("Stale patch: source snapshot does not match")
        return agent_code[:self.start] + self.after + agent_code[self.end:]

    def to_dict(self) -> dict[str, Any]:
        return {"operation": "REPLACE_PROMPT_SOURCE", **asdict(self), "evidence_refs": list(self.evidence_refs)}
