# """Promote a candidate only when its held-out metrics beat the champion.

# Promotion rule: candidate R2 must be strictly higher, while candidate MAE and
# MSE must be no worse. All metrics are calculated on the same log1p(view_count)
# scale by predict_model.py. If no champion alias exists, the first valid candidate
# becomes champion. A rejected candidate never changes the champion alias.
# """

# import os
# from pathlib import Path

# import mlflow
# from dotenv import load_dotenv
# from mlflow.exceptions import MlflowException
# from mlflow.tracking import MlflowClient

# from src.logger.logger import configure_logger

# logger = configure_logger()
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# REQUIRED_METRICS = ("mean_absolute_error", "mean_squared_error", "r2_score")


# def configure_mlflow() -> str:
#     load_dotenv(PROJECT_ROOT / ".env")
#     token = os.getenv("DAGSHUB_TOKEN")
#     owner = os.getenv("REPO_OWNER")
#     repo = os.getenv("REPO_NAME")
#     model_name = os.getenv("MODEL_NAME")
#     if not all([token, owner, repo, model_name]):
#         raise EnvironmentError(
#             "Missing DagsHub configuration. Set DAGSHUB_TOKEN, REPO_OWNER, REPO_NAME, and MODEL_NAME."
#         )

#     uri = f"https://dagshub.com/{owner}/{repo}.mlflow"
#     os.environ["MLFLOW_TRACKING_USERNAME"] = owner
#     os.environ["MLFLOW_TRACKING_PASSWORD"] = token
#     mlflow.set_tracking_uri(uri)
#     mlflow.set_registry_uri(uri)
#     logger.info("Using MLflow registry: %s", uri)
#     return model_name


# def get_alias_version(client: MlflowClient, model_name: str, alias: str):
#     try:
#         return client.get_model_version_by_alias(model_name, alias)
#     except MlflowException as exc:
#         # Treat only a genuinely missing alias as no existing champion.
#         # Authentication, network, and other registry errors must fail the job.
#         message = str(exc).lower()
#         if (
#             "not found" in message
#             or "does not exist" in message
#             or "resource_does_not_exist" in message
#         ):
#             return None
#         raise


# def get_metrics(client: MlflowClient, version, alias: str) -> dict:
#     if not version.run_id:
#         raise RuntimeError(
#             f"Model version {version.version} ({alias}) has no source run ID"
#         )
#     run = client.get_run(version.run_id)
#     metrics = run.data.metrics
#     missing = [key for key in REQUIRED_METRICS if key not in metrics]
#     if missing:
#         raise RuntimeError(
#             f"Model version {version.version} is missing comparison metrics: {missing}. "
#             "Do not promote until the candidate/champion has comparable evaluation metrics."
#         )
#     return {key: float(metrics[key]) for key in REQUIRED_METRICS}


# def candidate_is_better(candidate: dict, champion: dict) -> bool:
#     # Conservative rule: improve R2 and do not regress either error metric.
#     return (
#         candidate["r2_score"] > champion["r2_score"]
#         and candidate["mean_absolute_error"] <= champion["mean_absolute_error"]
#         and candidate["mean_squared_error"] <= champion["mean_squared_error"]
#     )


# def promote_model() -> bool:
#     model_name = configure_mlflow()
#     client = MlflowClient()

#     candidate = get_alias_version(client, model_name, "candidate")
#     if candidate is None:
#         raise RuntimeError(
#             f"No '{model_name}' model has the 'candidate' alias. Run DVC model registration first."
#         )
#     candidate_metrics = get_metrics(client, candidate, "candidate")
#     logger.info(
#         "Candidate version %s metrics: %s", candidate.version, candidate_metrics
#     )

#     champion = get_alias_version(client, model_name, "champion")
#     if champion is None:
#         client.set_registered_model_alias(
#             name=model_name, alias="champion", version=str(candidate.version)
#         )
#         logger.info(
#             "No existing champion. Candidate version %s is now champion.",
#             candidate.version,
#         )
#         return True

#     if str(champion.version) == str(candidate.version):
#         logger.info(
#             "Candidate version %s is already champion; no change needed.",
#             candidate.version,
#         )
#         return False

#     champion_metrics = get_metrics(client, champion, "champion")
#     logger.info(
#         "Current champion version %s metrics: %s", champion.version, champion_metrics
#     )

#     if candidate_is_better(candidate_metrics, champion_metrics):
#         client.set_registered_model_alias(
#             name=model_name, alias="champion", version=str(candidate.version)
#         )
#         logger.info(
#             "Promoted candidate version %s to champion; previous champion was %s.",
#             candidate.version,
#             champion.version,
#         )
#         return True

#     logger.info(
#         "Candidate version %s did not beat champion version %s. Champion remains unchanged. "
#         "Candidate metrics=%s; champion metrics=%s",
#         candidate.version,
#         champion.version,
#         candidate_metrics,
#         champion_metrics,
#     )
#     return False


# if __name__ == "__main__":
#     promote_model()

"""Promote a candidate only when its held-out metrics beat the champion.

Promotion rule: candidate R2 must be strictly higher, while candidate MAE and
MSE must be no worse. All metrics are calculated on the same log1p(view_count)
scale by predict_model.py. If no champion alias exists, the first valid candidate
becomes champion. A rejected candidate never changes the champion alias.
"""

import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.logger.logger import configure_logger

logger = configure_logger()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_METRICS = ("mean_absolute_error", "mean_squared_error", "r2_score")


def configure_mlflow() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    token = os.getenv("DAGSHUB_TOKEN")
    owner = os.getenv("REPO_OWNER")
    repo = os.getenv("REPO_NAME")
    model_name = str(os.getenv("MODEL_NAME"))
    if not all([token, owner, repo, model_name]):
        raise EnvironmentError(
            "Missing DagsHub configuration. Set DAGSHUB_TOKEN, REPO_OWNER, REPO_NAME, and MODEL_NAME."
        )

    uri = f"https://dagshub.com/{owner}/{repo}.mlflow"
    os.environ["MLFLOW_TRACKING_USERNAME"] = owner
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    logger.info("Using MLflow registry: %s", uri)
    return model_name


def get_alias_version(client: MlflowClient, model_name: str, alias: str):
    """Resolve an alias through the registered-model metadata.

    Reading the alias mapping first avoids calling the alias endpoint for a
    missing alias, which some registry backends report as INVALID_PARAMETER_VALUE.
    """
    try:
        registered_model = client.get_registered_model(model_name)
    except MlflowException as exc:
        message = str(exc).lower()
        if (
            "resource_does_not_exist" in message
            or "does not exist" in message
            or "not found" in message
        ):
            raise RuntimeError(
                f"Registered model '{model_name}' does not exist. "
                "Check MODEL_NAME and run the registration stage first."
            ) from exc
        raise

    aliases = getattr(registered_model, "aliases", None)
    if aliases is None:
        raise RuntimeError(
            f"The registry response for '{model_name}' does not expose aliases. "
            "Check the installed MLflow version and DagsHub registry compatibility."
        )

    if alias not in aliases:
        logger.info(
            "Alias '%s' is not set for registered model '%s'.",
            alias,
            model_name,
        )
        return None

    version = aliases[alias]
    return client.get_model_version(name=model_name, version=str(version))


def get_metrics(client: MlflowClient, version, alias: str) -> dict:
    if not version.run_id:
        raise RuntimeError(
            f"Model version {version.version} ({alias}) has no source run ID"
        )
    run = client.get_run(version.run_id)
    metrics = run.data.metrics
    missing = [key for key in REQUIRED_METRICS if key not in metrics]
    if missing:
        raise RuntimeError(
            f"Model version {version.version} is missing comparison metrics: {missing}. "
            "Do not promote until the candidate/champion has comparable evaluation metrics."
        )
    return {key: float(metrics[key]) for key in REQUIRED_METRICS}


def candidate_is_better(candidate: dict, champion: dict) -> bool:
    # Conservative rule: improve R2 and do not regress either error metric.
    return (
        candidate["r2_score"] > champion["r2_score"]
        and candidate["mean_absolute_error"] <= champion["mean_absolute_error"]
        and candidate["mean_squared_error"] <= champion["mean_squared_error"]
    )


def promote_model() -> bool:
    model_name = configure_mlflow()
    client = MlflowClient()

    candidate = get_alias_version(client, model_name, "candidate")
    if candidate is None:
        raise RuntimeError(
            f"No '{model_name}' model has the 'candidate' alias. Run DVC model registration first."
        )
    candidate_metrics = get_metrics(client, candidate, "candidate")
    logger.info(
        "Candidate version %s metrics: %s", candidate.version, candidate_metrics
    )

    champion = get_alias_version(client, model_name, "champion")
    if champion is None:
        client.set_registered_model_alias(
            name=model_name, alias="champion", version=str(candidate.version)
        )
        logger.info(
            "No existing champion. Candidate version %s is now champion.",
            candidate.version,
        )
        return True

    if str(champion.version) == str(candidate.version):
        logger.info(
            "Candidate version %s is already champion; no change needed.",
            candidate.version,
        )
        return False

    champion_metrics = get_metrics(client, champion, "champion")
    logger.info(
        "Current champion version %s metrics: %s", champion.version, champion_metrics
    )

    if candidate_is_better(candidate_metrics, champion_metrics):
        client.set_registered_model_alias(
            name=model_name, alias="champion", version=str(candidate.version)
        )
        logger.info(
            "Promoted candidate version %s to champion; previous champion was %s.",
            candidate.version,
            champion.version,
        )
        return True

    logger.info(
        "Candidate version %s did not beat champion version %s. Champion remains unchanged. "
        "Candidate metrics=%s; champion metrics=%s",
        candidate.version,
        champion.version,
        candidate_metrics,
        champion_metrics,
    )
    return False


if __name__ == "__main__":
    promote_model()
