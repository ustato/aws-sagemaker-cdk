import os
import tempfile

import pandas as pd
from sagemaker import image_uris
from sagemaker.estimator import Estimator
from sagemaker.inputs import CreateModelInput
from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.workflow.condition_step import ConditionStep, JsonGet
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import LocalPipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import CreateModelStep, ProcessingStep, TrainingStep
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# model_approval_status = ParameterString(
#     name="ModelApprovalStatus", default_value="PendingManualApproval"
# )

training_instance_type = "ml.m5.xlarge"

# Retrieve the docker image
train_image_uri = image_uris.retrieve(
    region="ap-northeast-1",
    framework="xgboost",
    version="1.0-1",
    py_version="py3",
    instance_type=training_instance_type,
)


# For local training a dummy role will be sufficient
role = "arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001"

tmpdir = tempfile.TemporaryDirectory()
data = load_boston()

X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.25, random_state=45
)
X_val, X_test, y_val, y_test = train_test_split(
    X_test, y_test, test_size=0.5, random_state=45
)

trainX = pd.DataFrame(X_train, columns=data.feature_names)
trainX["target"] = y_train

valX = pd.DataFrame(X_test, columns=data.feature_names)
valX["target"] = y_test

testX = pd.DataFrame(X_test, columns=data.feature_names)

os.makedirs(f"{tmpdir.name}/train/", exist_ok=True)
os.makedirs(f"{tmpdir.name}/validation/", exist_ok=True)
os.makedirs(f"{tmpdir.name}/test/", exist_ok=True)
local_train = f"{tmpdir.name}/train/boston_train.csv"
local_validation = f"{tmpdir.name}/validation/boston_validation.csv"
local_test = f"{tmpdir.name}/test/boston_test.csv"

trainX.to_csv(local_train, header=None, index=False)
valX.to_csv(local_validation, header=None, index=False)
testX.to_csv(local_test, header=None, index=False)


pipeline_session = LocalPipelineSession()

xgb_train = Estimator(
    image_uri=train_image_uri,
    instance_type="local",
    instance_count=1,
    role=role,
    hyperparameters={
        "num_round": 50,
        #     "max_depth": 5,
        #     "eta": 0.2,
        #     "gamma": 4,
        #     "min_child_weight": 6,
        #     "subsample": 0.7,
        #     "silent": 0,
    },
)

train_location = "file://" + local_train
validation_location = "file://" + local_validation


step_train = TrainingStep(
    name="MyTrainingStep",
    estimator=xgb_train,
    inputs={
        "train": train_location,
        "validation": validation_location,
    },
)

# script_eval = ScriptProcessor(
#     image_uri=train_image_uri,
#     command=["python3"],
#     instance_type="local",
#     instance_count=1,
#     base_job_name="script-abalone-eval",
#     role=role,
# )

# evaluation_report = PropertyFile(
#     name="EvaluationReport", output_name="evaluation", path="evaluation.json"
# )

# step_eval = ProcessingStep(
#     name="AbaloneEval",
#     processor=script_eval,
#     inputs=[
#         ProcessingInput(
#             source=local_train,
#             destination="/opt/ml/processing/model",
#         ),
#         ProcessingInput(
#             source=local_test,
#             destination="/opt/ml/processing/test",
#         ),
#     ],
#     outputs=[
#         ProcessingOutput(
#             output_name="evaluation", source="/opt/ml/processing/evaluation"
#         ),
#     ],
#     code="./evaluation.py",
#     property_files=[evaluation_report],
#     depends_on=[step_train],
# )

# model = Model(
#     image_uri=train_image_uri,
#     model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
#     sagemaker_session=pipeline_session,
#     role=role,
# )
# inputs = CreateModelInput(
#     instance_type="ml.m5.large",
#     accelerator_type="ml.eia1.medium",
# )
# step_create_model = CreateModelStep(
#     name="AbaloneCreateModel",
#     model=model,
#     inputs=inputs,
# )


# model_metrics = ModelMetrics(
#     model_statistics=MetricsSource(
#         s3_uri="{}/evaluation.json".format(
#             step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"][
#                 "S3Uri"
#             ]
#         ),
#         content_type="application/json",
#     )
# )
# step_register = RegisterModel(
#     name="AbaloneRegisterModel",
#     estimator=xgb_train,
#     model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
#     content_types=["text/csv"],
#     response_types=["text/csv"],
#     inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
#     transform_instances=["ml.m5.xlarge"],
#     model_package_group_name="hogehoge",
#     approval_status=model_approval_status,
#     model_metrics=model_metrics,
# )

# cond_lte = ConditionLessThanOrEqualTo(
#     left=JsonGet(
#         step=step_eval,
#         property_file=evaluation_report,
#         json_path="regression_metrics.mse.value",
#     ),
#     right=6.0,
# )

# step_cond = ConditionStep(
#     name="AbaloneMSECond",
#     conditions=[cond_lte],
#     if_steps=[
#         # step_register,
#         step_create_model,
#     ],
#     else_steps=[],
# )


pipeline = Pipeline(
    name="MyPipeline",
    steps=[
        step_train,
        # step_eval,
        # step_cond,
    ],
    sagemaker_session=pipeline_session,
)

pipeline.create(role_arn=role, description="local pipeline example")

execution = pipeline.start()

steps = execution.list_steps()

if steps["PipelineExecutionSteps"][0]["StepStatus"] == "Failed":
    print(steps["PipelineExecutionSteps"][0]["FailureReason"])
elif steps["PipelineExecutionSteps"][0]["StepStatus"] == "Succeeded":
    training_job_name = steps["PipelineExecutionSteps"][0]["Metadata"]["TrainingJob"][
        "Arn"
    ]
    step_outputs = pipeline_session.sagemaker_client.describe_training_job(
        TrainingJobName=training_job_name
    )
else:
    print(steps)
