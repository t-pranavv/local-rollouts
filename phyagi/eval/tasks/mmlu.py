# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from evaluate import load
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase

from phyagi.eval.distributed_utils import (
    all_gather_list,
    all_reduce_dict,
    is_main_process,
)
from phyagi.eval.generation import generate
from phyagi.utils.file_utils import save_json_file


class _MMLU:
    """MMLU evaluation task.

    Reference:
        Measuring Massive Multitask Language Understanding.
        https://arxiv.org/abs/2009.03300.

    """

    DATASET_PATH = "cais/mmlu"

    @staticmethod
    def mapping_fn(example: Dict[str, Any], subject: str) -> List[Dict[str, Any]]:
        subject = subject.replace("_", " ")
        description = f"The following are multiple choice questions (with answers) about {subject}."
        question = example["question"].strip()

        keys = ["A", "B", "C", "D"]
        choices = "".join([f"{key}. {choice}\n" for key, choice in zip(keys, example["choices"])])

        return [
            {
                "text": f"{description}\n\n{question}\n{choices}Answer: {key}",
                "target": key,
                "label": example["answer"],
            }
            for key in keys
        ]

    @staticmethod
    def run(
        dataset_name: str,
        model: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        device: Optional[Union[int, torch.device]] = None,
        n_examples: Optional[int] = None,
        output_file_path: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        metric = {
            "accuracy": load("accuracy"),
            "accuracy_norm": load("accuracy"),
        }

        dataset = load_dataset(_MMLU.DATASET_PATH, name=dataset_name)["test"]
        dataset = dataset.select(range(n_examples)) if n_examples is not None else dataset

        responses = generate(
            dataset,
            generation_engine="log_likelihood_pipeline",
            model=model,
            tokenizer=tokenizer,
            device=device,
            example_generator_kwargs={"mapping_fn": _MMLU.mapping_fn, "subject": dataset_name},
            **kwargs,
        )

        outputs = []
        for r in responses:
            log_likelihoods = r["log_likelihoods"]
            target_lengths = r["target_lengths"]
            label = r["labels"][0]

            prediction = np.argmax(log_likelihoods)
            prediction_norm = np.argmax(np.array(log_likelihoods) / target_lengths)

            metric["accuracy"].add(predictions=prediction, reference=label)
            metric["accuracy_norm"].add(predictions=prediction_norm, reference=label)

            outputs.append(r)

        results = {key: metric.compute()["accuracy"] for key, metric in metric.items()}

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

            outputs = all_gather_list(outputs)
            results = all_reduce_dict(results)

        if is_main_process():
            save_json_file(outputs, output_file_path) if output_file_path else None

        return results


class AbstractAlgebra(_MMLU):
    DATASET_NAME = "abstract_algebra"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(AbstractAlgebra.DATASET_NAME, *args, **kwargs)


class Anatomy(_MMLU):
    DATASET_NAME = "anatomy"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(Anatomy.DATASET_NAME, *args, **kwargs)


class Astronomy(_MMLU):
    DATASET_NAME = "astronomy"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            Astronomy.DATASET_NAME,
            *args,
            **kwargs,
        )


class BusinessEthics(_MMLU):
    DATASET_NAME = "business_ethics"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            BusinessEthics.DATASET_NAME,
            *args,
            **kwargs,
        )


class ClinicalKnowledge(_MMLU):
    DATASET_NAME = "clinical_knowledge"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            ClinicalKnowledge.DATASET_NAME,
            *args,
            **kwargs,
        )


class CollegeBiology(_MMLU):
    DATASET_NAME = "college_biology"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            CollegeBiology.DATASET_NAME,
            *args,
            **kwargs,
        )


class CollegeChemistry(_MMLU):
    DATASET_NAME = "college_chemistry"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            CollegeChemistry.DATASET_NAME,
            *args,
            **kwargs,
        )


class CollegeComputerScience(_MMLU):
    DATASET_NAME = "college_computer_science"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            CollegeComputerScience.DATASET_NAME,
            *args,
            **kwargs,
        )


class CollegeMathematics(_MMLU):
    DATASET_NAME = "college_mathematics"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            CollegeMathematics.DATASET_NAME,
            *args,
            **kwargs,
        )


class CollegeMedicine(_MMLU):
    DATASET_NAME = "college_medicine"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            CollegeMedicine.DATASET_NAME,
            *args,
            **kwargs,
        )


class CollegePhysics(_MMLU):
    DATASET_NAME = "college_physics"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            CollegePhysics.DATASET_NAME,
            *args,
            **kwargs,
        )


class ComputerSecurity(_MMLU):
    DATASET_NAME = "computer_security"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            ComputerSecurity.DATASET_NAME,
            *args,
            **kwargs,
        )


class ConceptualPhysics(_MMLU):
    DATASET_NAME = "conceptual_physics"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            ConceptualPhysics.DATASET_NAME,
            *args,
            **kwargs,
        )


class Econometrics(_MMLU):
    DATASET_NAME = "econometrics"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            Econometrics.DATASET_NAME,
            *args,
            **kwargs,
        )


class ElectricalEngineering(_MMLU):
    DATASET_NAME = "electrical_engineering"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            ElectricalEngineering.DATASET_NAME,
            *args,
            **kwargs,
        )


class ElementaryMathematics(_MMLU):
    DATASET_NAME = "elementary_mathematics"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            ElementaryMathematics.DATASET_NAME,
            *args,
            **kwargs,
        )


class FormalLogic(_MMLU):
    DATASET_NAME = "formal_logic"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            FormalLogic.DATASET_NAME,
            *args,
            **kwargs,
        )


class GlobalFacts(_MMLU):
    DATASET_NAME = "global_facts"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            GlobalFacts.DATASET_NAME,
            *args,
            **kwargs,
        )


class HighSchoolBiology(_MMLU):
    DATASET_NAME = "high_school_biology"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            HighSchoolBiology.DATASET_NAME,
            *args,
            **kwargs,
        )


class HighSchoolChemistry(_MMLU):
    DATASET_NAME = "high_school_chemistry"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            HighSchoolChemistry.DATASET_NAME,
            *args,
            **kwargs,
        )


class HighSchoolComputerScience(_MMLU):
    DATASET_NAME = "high_school_computer_science"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            HighSchoolComputerScience.DATASET_NAME,
            *args,
            **kwargs,
        )


class HighSchoolEuropeanHistory(_MMLU):
    DATASET_NAME = "high_school_european_history"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            HighSchoolEuropeanHistory.DATASET_NAME,
            *args,
            **kwargs,
        )


class HighSchoolGeography(_MMLU):
    DATASET_NAME = "high_school_geography"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            HighSchoolGeography.DATASET_NAME,
            *args,
            **kwargs,
        )


class HighSchoolGovernmentAndPolitics(_MMLU):
    DATASET_NAME = "high_school_government_and_politics"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            HighSchoolGovernmentAndPolitics.DATASET_NAME,
            *args,
            **kwargs,
        )


class HighSchoolMacroeconomics(_MMLU):
    DATASET_NAME = "high_school_macroeconomics"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            HighSchoolMacroeconomics.DATASET_NAME,
            *args,
            **kwargs,
        )


class HighSchoolMathematics(_MMLU):
    DATASET_NAME = "high_school_mathematics"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            HighSchoolMathematics.DATASET_NAME,
            *args,
            **kwargs,
        )


class HighSchoolMicroeconomics(_MMLU):
    DATASET_NAME = "high_school_microeconomics"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            HighSchoolMicroeconomics.DATASET_NAME,
            *args,
            **kwargs,
        )


class HighSchoolPhysics(_MMLU):
    DATASET_NAME = "high_school_physics"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            HighSchoolPhysics.DATASET_NAME,
            *args,
            **kwargs,
        )


class HighSchoolPsychology(_MMLU):
    DATASET_NAME = "high_school_psychology"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            HighSchoolPsychology.DATASET_NAME,
            *args,
            **kwargs,
        )


class HighSchoolStatistics(_MMLU):
    DATASET_NAME = "high_school_statistics"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            HighSchoolStatistics.DATASET_NAME,
            *args,
            **kwargs,
        )


class HighSchoolUSHistory(_MMLU):
    DATASET_NAME = "high_school_us_history"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            HighSchoolUSHistory.DATASET_NAME,
            *args,
            **kwargs,
        )


class HighSchoolWorldHistory(_MMLU):
    DATASET_NAME = "high_school_world_history"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            HighSchoolWorldHistory.DATASET_NAME,
            *args,
            **kwargs,
        )


class HumanAging(_MMLU):
    DATASET_NAME = "human_aging"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            HumanAging.DATASET_NAME,
            *args,
            **kwargs,
        )


class HumanSexuality(_MMLU):
    DATASET_NAME = "human_sexuality"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            HumanSexuality.DATASET_NAME,
            *args,
            **kwargs,
        )


class InternationalLaw(_MMLU):
    DATASET_NAME = "international_law"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            InternationalLaw.DATASET_NAME,
            *args,
            **kwargs,
        )


class Jurisprudence(_MMLU):
    DATASET_NAME = "jurisprudence"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            Jurisprudence.DATASET_NAME,
            *args,
            **kwargs,
        )


class LogicalFallacies(_MMLU):
    DATASET_NAME = "logical_fallacies"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            LogicalFallacies.DATASET_NAME,
            *args,
            **kwargs,
        )


class MachineLearning(_MMLU):
    DATASET_NAME = "machine_learning"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            MachineLearning.DATASET_NAME,
            *args,
            **kwargs,
        )


class Management(_MMLU):
    DATASET_NAME = "management"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            Management.DATASET_NAME,
            *args,
            **kwargs,
        )


class Marketing(_MMLU):
    DATASET_NAME = "marketing"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            Marketing.DATASET_NAME,
            *args,
            **kwargs,
        )


class MedicalGenetics(_MMLU):
    DATASET_NAME = "medical_genetics"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            MedicalGenetics.DATASET_NAME,
            *args,
            **kwargs,
        )


class Miscellaneous(_MMLU):
    DATASET_NAME = "miscellaneous"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            Miscellaneous.DATASET_NAME,
            *args,
            **kwargs,
        )


class MoralDisputes(_MMLU):
    DATASET_NAME = "moral_disputes"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            MoralDisputes.DATASET_NAME,
            *args,
            **kwargs,
        )


class MoralScenarios(_MMLU):
    DATASET_NAME = "moral_scenarios"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            MoralScenarios.DATASET_NAME,
            *args,
            **kwargs,
        )


class Nutrition(_MMLU):
    DATASET_NAME = "nutrition"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            Nutrition.DATASET_NAME,
            *args,
            **kwargs,
        )


class Philosophy(_MMLU):
    DATASET_NAME = "philosophy"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            Philosophy.DATASET_NAME,
            *args,
            **kwargs,
        )


class PreHistory(_MMLU):
    DATASET_NAME = "prehistory"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            PreHistory.DATASET_NAME,
            *args,
            **kwargs,
        )


class ProfessionalAccounting(_MMLU):
    DATASET_NAME = "professional_accounting"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            ProfessionalAccounting.DATASET_NAME,
            *args,
            **kwargs,
        )


class ProfessionalLaw(_MMLU):
    DATASET_NAME = "professional_law"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            ProfessionalLaw.DATASET_NAME,
            *args,
            **kwargs,
        )


class ProfessionalMedicine(_MMLU):
    DATASET_NAME = "professional_medicine"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            ProfessionalMedicine.DATASET_NAME,
            *args,
            **kwargs,
        )


class ProfessionalPsychology(_MMLU):
    DATASET_NAME = "professional_psychology"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            ProfessionalPsychology.DATASET_NAME,
            *args,
            **kwargs,
        )


class PublicRelations(_MMLU):
    DATASET_NAME = "public_relations"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            PublicRelations.DATASET_NAME,
            *args,
            **kwargs,
        )


class SecurityStudies(_MMLU):
    DATASET_NAME = "security_studies"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            SecurityStudies.DATASET_NAME,
            *args,
            **kwargs,
        )


class Sociology(_MMLU):
    DATASET_NAME = "sociology"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            Sociology.DATASET_NAME,
            *args,
            **kwargs,
        )


class USForeignPolicy(_MMLU):
    DATASET_NAME = "us_foreign_policy"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            USForeignPolicy.DATASET_NAME,
            *args,
            **kwargs,
        )


class Virology(_MMLU):
    DATASET_NAME = "virology"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            Virology.DATASET_NAME,
            *args,
            **kwargs,
        )


class WorldReligions(_MMLU):
    DATASET_NAME = "world_religions"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _MMLU.run(
            WorldReligions.DATASET_NAME,
            *args,
            **kwargs,
        )
