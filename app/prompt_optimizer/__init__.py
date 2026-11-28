"""External, agent-independent, trace-guided prompt optimization."""

from app.prompt_optimizer.diagnostic_report import load_diagnosis_reports, prompt_hypotheses
from app.prompt_optimizer.evaluation_report import EvaluationReport, load_evaluation_report
from app.prompt_optimizer.generator import LLMPatchGenerator, PromptModelPort
from app.prompt_optimizer.models import PromptComponent, PromptLesson, PromptPatch, PromptState, digest
from app.prompt_optimizer.optimizer import propose_candidate
from app.prompt_optimizer.policy import PromptGuard, SelectionConfig, SelectionPolicy
from app.prompt_optimizer.renderer import PromptRenderer, RenderedPromptBundle
from app.prompt_optimizer.service import OptimizationResult, PromptOptimizer
from app.prompt_optimizer.source_generator import LLMSourcePatchGenerator, SourcePatchGenerator
from app.prompt_optimizer.source_patch import SourcePromptPatch

__all__ = [
    "load_diagnosis_reports", "prompt_hypotheses", "PromptOptimizer", "OptimizationResult",
    "LLMSourcePatchGenerator", "SourcePatchGenerator", "SourcePromptPatch",
    "EvaluationReport", "load_evaluation_report",
    "LLMPatchGenerator", "PromptModelPort",
    "PromptComponent", "PromptLesson", "PromptPatch", "PromptState", "digest",
    "propose_candidate", "PromptGuard", "SelectionConfig", "SelectionPolicy",
    "PromptRenderer", "RenderedPromptBundle",
]
