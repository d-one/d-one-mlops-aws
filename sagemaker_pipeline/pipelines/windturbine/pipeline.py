# ================================================================================
# Author:      Heiko Kromer @ D ONE - 2022
# Description: This script contains the pipeline logic.
# ================================================================================
import os
import sys
import boto3
import sagemaker
import sagemaker.session
from datetime import datetime
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.condition_step import ConditionStep, JsonGet
from sagemaker.workflow.conditions import ConditionGreaterThan
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.functions import Join


from typing import List
 
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# ----------------- #
# --- Constants --- #
# ----------------- #

# Instances
PROCESSING_INSTANCE = "ml.m5.xlarge"
TRAINING_INSTANCE = "ml.m5.xlarge"
TRANSFORM_INSTANCES = ["ml.m5.xlarge"]
INFERENCE_INSTANCES = ["ml.t2.medium", "ml.m5.large"]

# S3 Bucket where the data is stored
BUCKET_NAME = "sagemaker-done-mlops"
BUCKET = f's3://{BUCKET_NAME}'

# Raw data paths
RAW_DATA_FOLDER = 'data'
RAW_DATA_FILE = 'wind_turbines.csv'
RAW_DATA_PATH = os.path.join(BUCKET, RAW_DATA_FOLDER, RAW_DATA_FILE)

# Path where the processed objects will be stored
now = datetime.now() # get current time to ensure uniqueness of the output folders
PROCESSED_DATA_FOLDER = 'processed_' + now.strftime("%Y-%m-%d_%H%M_%S%f")
PROCESSED_DATA_PATH = os.path.join(BUCKET, PROCESSED_DATA_FOLDER)

# Paths for model train, validation, test split
TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, 'train.csv')
TRAIN_DATA_PATH_W_HEADER = os.path.join(PROCESSED_DATA_PATH, 'train_w_header.csv')
VALIDATION_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, 'validation.csv')
TEST_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, 'test.csv')
TEST_DATA_PATH_W_HEADER = os.path.join(PROCESSED_DATA_PATH, 'test_w_header.csv')


# Model package group name
MODEL_PACKAGE_GROUP_NAME = "WindTurbinePackageGroup"
# Pipeline name
PIPELINE_NAME = "WindTurbinePipeline"
# Job prefix, will be used to identify our project
BASE_JOB_PREFIX = 'WindTurbineError'
# Preprocessor job name
PREPROCESSOR_JOB_NAME = 'SKLearn-Preprocessor'
# Name for the processing step
PROCESSING_STEP_NAME = f'{BASE_JOB_PREFIX}Preprocessing'
# Training job name
TRAINING_JOB_NAME = 'XGBoost-training'
# Name for the training step
TRAINING_STEP_NAME = f'{BASE_JOB_PREFIX}Training'
# Training job name
EVALUATION_JOB_NAME = 'XGBoost-evaluation'
# Name for the evaluation step
EVALUATION_STEP_NAME = f'{BASE_JOB_PREFIX}Evaluation'
# Name for the model register step
REGISTER_MODEL_STEP_NAME = f'{BASE_JOB_PREFIX}RegisterModel'
# Name for the condition step
CONDITON_STEP_NAME = f'{BASE_JOB_PREFIX}AccuracyCondition'
# Name for the fail ste
FAIL_STEP_NAME = f'{BASE_JOB_PREFIX}Fail'
# Accuracy value that the model need to reach as minimum
ACCURACY_CONDITION  = 0.75

def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.
    
    Args:
        region: The aws region to start the session.
        default_bucket: The bucket to use for storing the artifacts.
        
    Returns:
        `sagemaker.session.Session` instance.
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


def get_sagemaker_client(region):
    """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
    """
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker_client


def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    """Gets the pipeline custom tags.
    
    Args:
        new_tags: Project tags.
        region: The aws region to start the session.
        sagemaker_project_arn: Amazon Resource Name.
        
    Returns:
        Tags.
    """
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region: str,
    role: str = None,
    default_bucket: str = None,
    sagemaker_project_arn: str = None,
    processing_instance: str = PROCESSING_INSTANCE,
    training_instance: str = TRAINING_INSTANCE,
    transform_instances: List[str] = TRANSFORM_INSTANCES,
    inference_instances: List[str] = INFERENCE_INSTANCES,
    model_package_group_name: str = MODEL_PACKAGE_GROUP_NAME, 
    pipeline_name: str = PIPELINE_NAME,
    base_job_prefix: str = BASE_JOB_PREFIX,
    preprocessor_job_name: str = PREPROCESSOR_JOB_NAME,
    processing_step_name: str = PROCESSING_STEP_NAME,
    training_job_name: str = TRAINING_JOB_NAME,
    training_step_name: str = TRAINING_STEP_NAME,
    evaluation_job_name: str = EVALUATION_JOB_NAME,
    evaluation_step_name: str = EVALUATION_STEP_NAME,
    register_model_step_name: str = REGISTER_MODEL_STEP_NAME,
    condition_step_name: str = CONDITON_STEP_NAME,
    fail_step_name: str = FAIL_STEP_NAME,
    accuracy_value: float = ACCURACY_CONDITION
    ):
    """Gets a SageMaker ML Pipeline instance working with windturbine data.
    
    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts.
        sagemaker_project_arn: ARN of the project.
        processing_instance: Instance type for processing.
        training_instance: Instance type for training.
        transform_instances: Instance type for transforming.
        inference_instances: Instance type for inference.
        pipeline_name: Name for the pipeline.
        base_job_prefix: Prefix for the pipeline job.
        preprocessor_job_name: Preprocessor job name.
        processing_step_name: Name for the preprocessing pipeline step.
        training_job_name: Name for the training job.
        training_step_name: Name for the training pipeline step.
        evaluation_job_name: Name for the evaluation job.
        evaluation_step_name: Name for the evaluation pipeline step.
        register_model_step_name: Name for the register model step.
        condition_step_name: Name for the step for the condition.
        fail_step_name: Name for the step when condition fails.
        accuracy_value: Minimum accuracy required for model to register. Used in condition step.
        
    Returns:
        An instance of a pipeline.
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)
    
    # Print the role for debugging
    print(f"SageMaker assumes role: {role}.")
        
    model_approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value="PendingManualApproval",  # We want manual approval
    )
    input_data = ParameterString(
        name="InputDataUrl",
        default_value=RAW_DATA_PATH,  
    )

    # Processing step for feature engineering
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type=training_instance,
        instance_count=1,
        base_job_name=f"{base_job_prefix}-{preprocessor_job_name}",  
        sagemaker_session=sagemaker_session,
        role=role,
    )
    step_process = ProcessingStep(
        name=processing_step_name,
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                source=RAW_DATA_PATH,
                destination="/opt/ml/processing/input"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                destination=PROCESSED_DATA_PATH,
                source="/opt/ml/processing/train"
            ),
            ProcessingOutput(
                output_name="validation",
                destination=PROCESSED_DATA_PATH,
                source="/opt/ml/processing/validation"
            ),
            ProcessingOutput(
                output_name="test",
                destination=PROCESSED_DATA_PATH,
                source='/opt/ml/processing/test'
            ),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        job_arguments=[
            "--n_test_days", "20",
            "--n_val_days", "30"
        ],
    )

    # Training step for generating model artifacts
    model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/AbaloneTrain"
    # Retrieving the pre-configured AWS Sagemaker xgboost algorithm
    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",  
        region=region,
        version="1.0-1",
        py_version="py3",
        instance_type=training_instance
    )
    xgboost_model = sagemaker.estimator.Estimator(
        image_uri=image_uri,
        instance_type=training_instance,
        instance_count=1,
        output_path=model_path,
        base_job_name=f"{base_job_prefix}-{training_job_name}",
        sagemaker_session=sagemaker_session,
        role=role,
    )
    xgboost_model.set_hyperparameters(
        objective="binary:logistic",
        num_round=50,
        max_depth=5,
        eta=0.2,
        gamma=4,
        min_child_weight=6,
        subsample=0.7,
        silent=0,
    )
    step_train = TrainingStep(
        name=training_step_name,
        estimator=xgboost_model,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )

    # Processing step for evaluation
    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=processing_instance,
        instance_count=1,
        base_job_name=f"{base_job_prefix}-{evaluation_job_name}",
        sagemaker_session=sagemaker_session,
        role=role,
    )
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    step_eval = ProcessingStep(
        name=evaluation_step_name,
        processor=script_eval,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                destination=PROCESSED_DATA_PATH,
                source="/opt/ml/processing/evaluation"
            ),
        ],
        code=os.path.join(BASE_DIR, "evaluate.py"),
        property_files=[evaluation_report],
    )

    # Register model step that will be conditionally executed
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json",
        )
    )

    # Register model step that will be conditionally executed
    step_register = RegisterModel(
        name=register_model_step_name,
        estimator=xgboost_model,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=inference_instances,
        transform_instances=transform_instances,
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )
    
    # Conditional step for evaluating model quality and branching execution
    cond_lte = ConditionGreaterThan(
        left=JsonGet(
            step=step_eval,
            property_file=evaluation_report,
            json_path="binary_classification_metrics.accuracy.value"
        ),
        right=accuracy_value
    )

    step_fail = FailStep(
        name=fail_step_name,
        error_message=Join(on=" ", values=["Execution failed due to accuracy <", accuracy_value]),
    )

    step_cond = ConditionStep(
        name=condition_step_name,
        conditions=[cond_lte],
        if_steps=[step_register],
        else_steps=[step_fail],
    )

    # Pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            model_approval_status,
            input_data,
        ],
        steps=[
            step_process,
            step_train,
            step_eval,
            step_cond,
        ],
        sagemaker_session=sagemaker_session,
    )
    return pipeline
