base_job_name = PREPROCESSING_JOB_NAME
sklearn_processor = SKLearnProcessor(
    base_job_name=base_job_name,
    framework_version="0.20.0",
    role=ROLE,
    instance_type="ml.m5.xlarge",
    instance_count=1
)

sklearn_processor.run(
    code="prepare_data.py",
    inputs=[
        ProcessingInput(
            source=RAW_DATA_PATH,
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        ProcessingOutput(
            destination=PROCESSED_DATA_PATH,
            source="/opt/ml/processing/train"
        ),
        ProcessingOutput(
            destination=PROCESSED_DATA_PATH,
            source="/opt/ml/processing/validation"
        ),
        ProcessingOutput(
            destination=PROCESSED_DATA_PATH,
            source="/opt/ml/processing/test"
        ),
    ],
    arguments=[
        "--n_test_days", "20",
        "--n_val_days", "30"
    ],
)

preprocessing_job_description = sklearn_processor.jobs[-1].describe()
