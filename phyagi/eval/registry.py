# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
from transformers import PreTrainedTokenizerBase

from phyagi.eval.tasks.agi_eval import (
    AGIMATH,
    LSATAR,
    LSATLR,
    LSATRC,
    AQuARAT,
    LogiQAEn,
    SATEn,
    SATMath,
)
from phyagi.eval.tasks.arc import ARCChallenge, ARCEasy
from phyagi.eval.tasks.arithmetic import (
    Arithmetic1DC,
    Arithmetic2DA,
    Arithmetic2DM,
    Arithmetic2DS,
    Arithmetic3DA,
    Arithmetic3DS,
    Arithmetic4DA,
    Arithmetic4DS,
    Arithmetic5DA,
    Arithmetic5DS,
)
from phyagi.eval.tasks.completion import Completion
from phyagi.eval.tasks.glue import MNLI, MRPC, QNLI, QQP, RTE, SST2, WNLI, CoLA
from phyagi.eval.tasks.gsm8k import GSM8K
from phyagi.eval.tasks.hellaswag import HellaSwag
from phyagi.eval.tasks.human_eval import HumanEval
from phyagi.eval.tasks.human_eval_pack import HumanEvalPackJS, HumanEvalPackPython
from phyagi.eval.tasks.lambada import LAMBADA, LAMBADAOpenAI
from phyagi.eval.tasks.loss import LossHFHubDataset, LossNumpyDataset
from phyagi.eval.tasks.math import MATH
from phyagi.eval.tasks.math_qa import MathQA
from phyagi.eval.tasks.mbpp import MBPP
from phyagi.eval.tasks.medmcqa import MedMCQA
from phyagi.eval.tasks.mmlu import (
    AbstractAlgebra,
    Anatomy,
    Astronomy,
    BusinessEthics,
    ClinicalKnowledge,
    CollegeBiology,
    CollegeChemistry,
    CollegeComputerScience,
    CollegeMathematics,
    CollegeMedicine,
    CollegePhysics,
    ComputerSecurity,
    ConceptualPhysics,
    Econometrics,
    ElectricalEngineering,
    ElementaryMathematics,
    FormalLogic,
    GlobalFacts,
    HighSchoolBiology,
    HighSchoolChemistry,
    HighSchoolComputerScience,
    HighSchoolEuropeanHistory,
    HighSchoolGeography,
    HighSchoolGovernmentAndPolitics,
    HighSchoolMacroeconomics,
    HighSchoolMathematics,
    HighSchoolMicroeconomics,
    HighSchoolPhysics,
    HighSchoolPsychology,
    HighSchoolStatistics,
    HighSchoolUSHistory,
    HighSchoolWorldHistory,
    HumanAging,
    HumanSexuality,
    InternationalLaw,
    Jurisprudence,
    LogicalFallacies,
    MachineLearning,
    Management,
    Marketing,
    MedicalGenetics,
    Miscellaneous,
    MoralDisputes,
    MoralScenarios,
    Nutrition,
    Philosophy,
    PreHistory,
    ProfessionalAccounting,
    ProfessionalLaw,
    ProfessionalMedicine,
    ProfessionalPsychology,
    PublicRelations,
    SecurityStudies,
    Sociology,
    USForeignPolicy,
    Virology,
    WorldReligions,
)
from phyagi.eval.tasks.natural_questions import NaturalQuestions
from phyagi.eval.tasks.openbookqa import OpenBookQAAdditional, OpenBookQAMain
from phyagi.eval.tasks.p3 import P3
from phyagi.eval.tasks.piqa import PIQA
from phyagi.eval.tasks.race import RACEHigh, RACEMiddle
from phyagi.eval.tasks.siqa import SIQA
from phyagi.eval.tasks.squad import SQuAD
from phyagi.eval.tasks.super_glue import (
    CB,
    COPA,
    WSC,
    AXb,
    AXg,
    BoolQ,
    MultiRC,
    ReCoRD,
    WiC,
)
from phyagi.eval.tasks.trivia_qa import TriviaQA
from phyagi.eval.tasks.winogrande import WinoGrande
from phyagi.utils.import_utils import is_lm_eval_available
from phyagi.utils.logging_utils import get_logger

LMEvaluationHarness = None
if is_lm_eval_available():
    from phyagi.eval.tasks.lm_evaluation_harness import LMEvaluationHarness

logger = get_logger(__name__)


TASKS = {
    "abstract_algebra": AbstractAlgebra,
    "agi_math": AGIMATH,
    "anatomy": Anatomy,
    "astronomy": Astronomy,
    "arc_easy": ARCEasy,
    "arc_challenge": ARCChallenge,
    "arithmetic_1dc": Arithmetic1DC,
    "arithmetic_2da": Arithmetic2DA,
    "arithmetic_2dm": Arithmetic2DM,
    "arithmetic_2ds": Arithmetic2DS,
    "arithmetic_3da": Arithmetic3DA,
    "arithmetic_3ds": Arithmetic3DS,
    "arithmetic_4da": Arithmetic4DA,
    "arithmetic_4ds": Arithmetic4DS,
    "arithmetic_5da": Arithmetic5DA,
    "arithmetic_5ds": Arithmetic5DS,
    "aqua_rat": AQuARAT,
    "axb": AXb,
    "axg": AXg,
    "boolq": BoolQ,
    "business_ethics": BusinessEthics,
    "cb": CB,
    "clinical_knowledge": ClinicalKnowledge,
    "cola": CoLA,
    "college_biology": CollegeBiology,
    "college_chemistry": CollegeChemistry,
    "college_computer_science": CollegeComputerScience,
    "college_mathematics": CollegeMathematics,
    "college_medicine": CollegeMedicine,
    "college_physics": CollegePhysics,
    "completion": Completion,
    "computer_security": ComputerSecurity,
    "conceptual_physics": ConceptualPhysics,
    "copa": COPA,
    "econometrics": Econometrics,
    "elementary_mathematics": ElementaryMathematics,
    "electrical_engineering": ElectricalEngineering,
    "formal_logic": FormalLogic,
    "global_facts": GlobalFacts,
    "gsm8k": GSM8K,
    "hellaswag": HellaSwag,
    "high_school_biology": HighSchoolBiology,
    "high_school_chemistry": HighSchoolChemistry,
    "high_school_computer_science": HighSchoolComputerScience,
    "high_school_european_history": HighSchoolEuropeanHistory,
    "high_school_geography": HighSchoolGeography,
    "high_school_government_and_politics": HighSchoolGovernmentAndPolitics,
    "high_school_macroeconomics": HighSchoolMacroeconomics,
    "high_school_mathematics": HighSchoolMathematics,
    "high_school_microeconomics": HighSchoolMicroeconomics,
    "high_school_physics": HighSchoolPhysics,
    "high_school_psychology": HighSchoolPsychology,
    "high_school_statistics": HighSchoolStatistics,
    "high_school_us_history": HighSchoolUSHistory,
    "high_school_world_history": HighSchoolWorldHistory,
    "human_aging": HumanAging,
    "human_eval": HumanEval,
    "human_eval_pack_python": HumanEvalPackPython,
    "human_eval_pack_js": HumanEvalPackJS,
    "human_sexuality": HumanSexuality,
    "international_law": InternationalLaw,
    "jurisprudence": Jurisprudence,
    "lambada": LAMBADA,
    "lambada_openai": LAMBADAOpenAI,
    "lm_eval": LMEvaluationHarness,
    "logical_fallacies": LogicalFallacies,
    "logiqa_en": LogiQAEn,
    "loss_hf_hub": LossHFHubDataset,
    "loss_numpy": LossNumpyDataset,
    "lsat_ar": LSATAR,
    "lsat_lr": LSATLR,
    "lsat_rc": LSATRC,
    "machine_learning": MachineLearning,
    "management": Management,
    "marketing": Marketing,
    "math": MATH,
    "math_qa": MathQA,
    "mbpp": MBPP,
    "medmcqa": MedMCQA,
    "medical_genetics": MedicalGenetics,
    "miscellaneous": Miscellaneous,
    "mnli": MNLI,
    "moral_disputes": MoralDisputes,
    "moral_scenarios": MoralScenarios,
    "mrpc": MRPC,
    "multirc": MultiRC,
    "natural_questions": NaturalQuestions,
    "nutrition": Nutrition,
    "openbookqaadditional": OpenBookQAAdditional,
    "openbookqa": OpenBookQAMain,
    "philosophy": Philosophy,
    "p3": P3,
    "piqa": PIQA,
    "pre_history": PreHistory,
    "professional_accounting": ProfessionalAccounting,
    "professional_law": ProfessionalLaw,
    "professional_medicine": ProfessionalMedicine,
    "professional_psychology": ProfessionalPsychology,
    "public_relations": PublicRelations,
    "qnli": QNLI,
    "qqp": QQP,
    "race_high": RACEHigh,
    "race_middle": RACEMiddle,
    "record": ReCoRD,
    "rte": RTE,
    "sat_en": SATEn,
    "sat_math": SATMath,
    "security_studies": SecurityStudies,
    "siqa": SIQA,
    "sociology": Sociology,
    "squad": SQuAD,
    "sst2": SST2,
    "trivia_qa": TriviaQA,
    "us_foreign_policy": USForeignPolicy,
    "virology": Virology,
    "wic": WiC,
    "winogrande": WinoGrande,
    "world_religions": WorldReligions,
    "wnli": WNLI,
    "wsc": WSC,
}


def run_task(
    task_name: str,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    device: Optional[Union[int, torch.device]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Run an evaluation task.

    This function wraps the ``run()`` method of the task class. It is a convenience function for
    running a task without having to import the task class directly.

    Since each task might have a different set of keyword arguments, this function takes keyword
    arguments and passes them directly to the ``run()`` method. However, the following arguments
    are required to all tasks: ``model`` and ``tokenizer``.

    Args:
        task_name: Name of the task.
        model: Instance of a model.
        tokenizer: Instance of a tokenizer.
        device: Device to use for running the task.

    """

    logger.info(f"Loading task: {task_name}")
    logger.info(f"Task configuration: {kwargs}")

    if task_name not in TASKS:
        raise ValueError(f"`task_name` must be one of {list(TASKS.keys())}, but got '{task_name}'.")

    task_cls = TASKS[task_name]
    if task_cls is None:
        raise ValueError(f"Task '{task_name}' is not available due to a missing dependency.")

    return task_cls.run(model, tokenizer, device=device, **kwargs)
