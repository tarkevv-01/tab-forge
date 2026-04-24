"""
LLM Runner — Self-Consistency Ranking
======================================

Sends a prompt N times to an OpenAI-compatible API, parses RANKING_START/END
blocks from each response, and aggregates results into average ranks.

Usage
-----
    from llm_runner import LLMRunner

    runner = LLMRunner(
        base_url = "https://api.openai.com/v1",
        api_key  = "sk-...",
        model    = "gpt-4o",
    )

    result = runner.run(
        prompt  = prompt,          # string from PromptGenerator
        n_runs  = 5,               # self-consistency iterations
        temperature = 0.7,
        top_p       = 0.9,
    )

    print(result.average_ranks)    # {'DDPM': 1.4, 'CTGAN': 2.6, ...}
    print(result)                  # full summary
"""

import re
import time
import statistics
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI


# ─────────────────────────────────────────────────────────────────────────────
# Output structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SingleRun:
    """Result of one LLM call."""
    run_index:   int
    raw_response: str
    ranking:     list[str]          # e.g. ['DDPM', 'CTGAN', ...], len=n_models
    parsed_ok:   bool               # False if RANKING block was missing/malformed


@dataclass
class RunnerResult:
    """Aggregated result across all n_runs."""
    runs:         list[SingleRun]
    average_ranks: dict[str, float]  # model → mean rank (1 = best)
    final_ranking: list[str]         # models sorted by average rank
    n_runs:       int
    n_parsed:     int                # how many runs were successfully parsed
    model:        str
    prompt_chars: int

    def __repr__(self) -> str:
        sep = "─" * 50
        lines = [
            f"LLMRunner result  |  model={self.model}  |  "
            f"runs={self.n_runs}  parsed={self.n_parsed}/{self.n_runs}",
            sep,
            "Average ranks (lower = better):",
        ]
        for i, m in enumerate(self.final_ranking, 1):
            lines.append(f"  {i}. {m:<12}  avg_rank={self.average_ranks[m]:.2f}")
        lines.append(sep)
        lines.append("Individual runs:")
        for r in self.runs:
            status = "OK" if r.parsed_ok else "PARSE ERROR"
            ranking_str = " > ".join(r.ranking) if r.ranking else "—"
            lines.append(f"  run {r.run_index+1:>2}  [{status}]  {ranking_str}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_ranking(text: str) -> list[str]:
    """
    Extract the ordered list of model names from a RANKING_START/END block.

    Returns a list of model names in ranked order (index 0 = rank 1),
    or an empty list if the block is absent or malformed.
    """
    match = re.search(
        r"RANKING_START\s*\n(.*?)\nRANKING_END",
        text,
        re.DOTALL,
    )
    if not match:
        return []

    block = match.group(1)
    models = re.findall(r"^\s*\d+\.\s*(.+?)\s*$", block, re.MULTILINE)
    return models


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

class LLMRunner:
    """
    Sends a prompt N times and aggregates rankings via self-consistency.

    Parameters
    ----------
    base_url : str
        OpenAI-compatible API base URL.
        Examples:
          "https://api.openai.com/v1"
          "http://localhost:11434/v1"   (Ollama)
          "https://api.together.xyz/v1"
    api_key : str
        API key. Pass a dummy string (e.g. "ollama") for local servers
        that don't require authentication.
    model : str
        Model identifier, e.g. "gpt-4o", "llama3", "mistral".
    retry_on_parse_error : bool
        If True, a run that fails to parse is retried once. Default False.
    request_delay : float
        Seconds to sleep between requests (useful for rate-limited APIs).
        Default 0.
    """

    def __init__(
        self,
        base_url: str,
        api_key:  str,
        model:    str,
        retry_on_parse_error: bool  = False,
        request_delay:        float = 0.0,
    ):
        self.model   = model
        self.retry   = retry_on_parse_error
        self.delay   = request_delay
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    # ─────────────────────────────────────────────────────────────────────────

    def run(
        self,
        prompt:      str,
        n_runs:      int   = 1,
        # — model sampling params —
        temperature: float = 0.7,
        top_p:       float = 1.0,
        max_tokens:  int   = 2048,
        presence_penalty:  float = 0.0,
        frequency_penalty: float = 0.0,
        seed:        Optional[int] = None,
    ) -> RunnerResult:
        """
        Run the prompt n_runs times and return aggregated rankings.

        Parameters
        ----------
        prompt : str
            The full prompt string (e.g. from PromptGenerator.build_prompt).
        n_runs : int
            Number of independent LLM calls (self-consistency iterations).
        temperature : float
            Sampling temperature. Higher = more varied rankings.
        top_p : float
            Nucleus sampling probability mass.
        max_tokens : int
            Maximum tokens in each response.
        presence_penalty : float
            OpenAI presence penalty (-2 to 2).
        frequency_penalty : float
            OpenAI frequency penalty (-2 to 2).
        seed : int, optional
            Fixed seed for reproducibility (supported by some APIs).

        Returns
        -------
        RunnerResult
        """
        single_runs: list[SingleRun] = []

        for i in range(n_runs):
            print(f"  run {i+1}/{n_runs}...", end=" ", flush=True)

            raw = self._call(
                prompt            = prompt,
                temperature       = temperature,
                top_p             = top_p,
                max_tokens        = max_tokens,
                presence_penalty  = presence_penalty,
                frequency_penalty = frequency_penalty,
                seed              = seed,
            )

            ranking   = _parse_ranking(raw)
            parsed_ok = bool(ranking)

            # optional single retry on parse failure
            if not parsed_ok and self.retry:
                print("parse error — retrying...", end=" ", flush=True)
                raw     = self._call(
                    prompt            = prompt,
                    temperature       = temperature,
                    top_p             = top_p,
                    max_tokens        = max_tokens,
                    presence_penalty  = presence_penalty,
                    frequency_penalty = frequency_penalty,
                    seed              = seed,
                )
                ranking   = _parse_ranking(raw)
                parsed_ok = bool(ranking)

            status = "OK" if parsed_ok else "PARSE ERROR"
            print(status)

            single_runs.append(SingleRun(
                run_index    = i,
                raw_response = raw,
                ranking      = ranking,
                parsed_ok    = parsed_ok,
            ))

            if self.delay and i < n_runs - 1:
                time.sleep(self.delay)

        return self._aggregate(single_runs, prompt)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _call(self, prompt: str, **sampling_kwargs) -> str:
        """Single API call; returns the raw response string."""
        kwargs = {k: v for k, v in sampling_kwargs.items() if v is not None}
        response = self._client.chat.completions.create(
            model    = self.model,
            messages = [{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response.choices[0].message.content or ""

    def _aggregate(
        self,
        runs:   list[SingleRun],
        prompt: str,
    ) -> RunnerResult:
        """Compute average rank per model across all successfully parsed runs."""
        parsed = [r for r in runs if r.parsed_ok]

        # Collect all model names seen across runs
        all_models: list[str] = []
        for r in parsed:
            for m in r.ranking:
                if m not in all_models:
                    all_models.append(m)

        average_ranks: dict[str, float] = {}

        if not parsed:
            # nothing parsed — return empty aggregation
            final_ranking = all_models
        else:
            rank_lists: dict[str, list[int]] = {m: [] for m in all_models}

            for r in parsed:
                for rank_idx, model_name in enumerate(r.ranking, start=1):
                    if model_name in rank_lists:
                        rank_lists[model_name].append(rank_idx)

            for m in all_models:
                ranks = rank_lists[m]
                average_ranks[m] = statistics.mean(ranks) if ranks else float("inf")

            final_ranking = sorted(all_models, key=lambda m: average_ranks.get(m, float("inf")))

        return RunnerResult(
            runs          = runs,
            average_ranks = average_ranks,
            final_ranking = final_ranking,
            n_runs        = len(runs),
            n_parsed      = len(parsed),
            model         = self.model,
            prompt_chars  = len(prompt),
        )
