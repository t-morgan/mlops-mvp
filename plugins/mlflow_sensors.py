from __future__ import annotations
from typing import Any

from airflow.sensors.base import BaseSensorOperator
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException


class MLflowModelVersionSensor(BaseSensorOperator):
    """
    Waits for a specific version of an MLflow model to be in a specific stage.

    :param model_name: The name of the model in the MLflow Model Registry.
    :param target_stage: The stage to wait for (e.g., 'Staging', 'Production').
    :param mlflow_conn_id: The Airflow connection ID for the MLflow tracking server.
                           (We will use environment variables instead for this MVP).
    """

    template_fields: tuple[str, ...] = ("model_name", "target_stage")

    def __init__(
        self,
        *,
        model_name: str,
        target_stage: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model_name = model_name
        self.target_stage = target_stage
        self.client = None

    def _get_mlflow_client(self) -> MlflowClient:
        """Initializes the MlflowClient. We reuse it to be efficient."""
        if not self.client:
            # The client will be configured by the MLFLOW_TRACKING_URI
            # environment variable set in the Airflow container.
            self.client = MlflowClient()
        return self.client

    def poke(self, context: Any) -> bool:
        """

        This is the function that Airflow calls every `poke_interval`.
        It should return True if the condition is met, and False otherwise.
        """
        client = self._get_mlflow_client()
        self.log.info(
            f"Poking for model '{self.model_name}' to be in stage '{self.target_stage}'."
        )
        try:
            # Get the latest versions of the model for the target stage
            latest_versions = client.get_latest_versions(
                name=self.model_name, stages=[self.target_stage]
            )

            if latest_versions:
                self.log.info(
                    f"Success! Found model '{self.model_name}' version "
                    f"'{latest_versions[0].version}' in stage '{self.target_stage}'."
                )
                # You could push the version number to XComs here if needed
                # context["ti"].xcom_push(key="model_version", value=latest_versions[0].version)
                return True
            else:
                self.log.info(
                    f"Model '{self.model_name}' not yet in stage '{self.target_stage}'. Waiting..."
                )
                return False

        except MlflowException as e:
            # This can happen if the model doesn't exist yet at all.
            self.log.warning(
                f"MLflow exception: {e}. The model may not be registered yet."
            )
            return False
