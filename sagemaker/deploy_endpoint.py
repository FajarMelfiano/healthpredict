"""
deploy_endpoint.py
Deploy an Approved model from SageMaker Model Registry to a real-time endpoint.

AWS Academy Learner Lab Notes:
  - Uses ml.t2.medium — cheapest inference instance
  - ALWAYS delete endpoint after testing to preserve credits
  - Run this AFTER approving the model in Model Registry

Usage:
  python deploy_endpoint.py
"""

import boto3
import sagemaker
import logging
import time

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ── CONFIG ──────────────────────────────────────────────────
STUDENT_NAME   = "yourname"          # ← CHANGE THIS
ACCOUNT_ID     = boto3.client('sts').get_caller_identity()['Account']
REGION         = "us-east-1"
ROLE_ARN       = f"arn:aws:iam::{ACCOUNT_ID}:role/LabRole"

PROJECT        = "healthpredict"
MODEL_GROUP    = f"{PROJECT}-model-group"
ENDPOINT_NAME  = f"{PROJECT}-endpoint-diabetes"
CONFIG_NAME    = f"{PROJECT}-endpoint-config"
MODEL_NAME     = f"{PROJECT}-model-diabetes"

sm = boto3.client('sagemaker', region_name=REGION)
sm_runtime = boto3.client('sagemaker-runtime', region_name=REGION)


def get_latest_approved_model_arn() -> str:
    """Retrieve the ARN of the latest Approved model package."""
    resp = sm.list_model_packages(
        ModelPackageGroupName=MODEL_GROUP,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    packages = resp.get('ModelPackageSummaryList', [])
    if not packages:
        raise RuntimeError(
            f"No Approved model found in group '{MODEL_GROUP}'. "
            "Please approve a model in SageMaker Model Registry first."
        )
    arn = packages[0]['ModelPackageArn']
    logger.info(f"Found approved model: {arn}")
    return arn


def create_model(model_package_arn: str) -> None:
    """Create a SageMaker Model from the approved model package."""
    # Clean up existing model
    try:
        sm.delete_model(ModelName=MODEL_NAME)
        logger.info(f"Deleted existing model: {MODEL_NAME}")
    except sm.exceptions.ClientError:
        pass

    sm.create_model(
        ModelName=MODEL_NAME,
        ExecutionRoleArn=ROLE_ARN,
        Containers=[{
            "ModelPackageName": model_package_arn,
        }],
        Tags=[{"Key": "Project", "Value": PROJECT}]
    )
    logger.info(f"Model created: {MODEL_NAME}")


def create_endpoint_config() -> None:
    """Create endpoint configuration using ml.t2.medium (cost-efficient)."""
    try:
        sm.delete_endpoint_config(EndpointConfigName=CONFIG_NAME)
        logger.info(f"Deleted existing endpoint config: {CONFIG_NAME}")
    except sm.exceptions.ClientError:
        pass

    sm.create_endpoint_config(
        EndpointConfigName=CONFIG_NAME,
        ProductionVariants=[{
            "VariantName":          "AllTraffic",
            "ModelName":            MODEL_NAME,
            "InitialInstanceCount": 1,
            "InstanceType":         "ml.t2.medium",  # cheapest option for AWS Academy
            "InitialVariantWeight": 1.0,
        }],
        Tags=[{"Key": "Project", "Value": PROJECT}]
    )
    logger.info(f"Endpoint config created: {CONFIG_NAME}")


def deploy_endpoint() -> None:
    """Create or update the SageMaker endpoint."""
    try:
        desc = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        status = desc['EndpointStatus']
        logger.info(f"Endpoint exists with status: {status}")
        if status in ('InService', 'OutOfService', 'Failed'):
            sm.update_endpoint(
                EndpointName=ENDPOINT_NAME,
                EndpointConfigName=CONFIG_NAME,
            )
            logger.info("Updating existing endpoint...")
        else:
            logger.warning(f"Endpoint in transitional state: {status}. Wait before retrying.")
            return
    except sm.exceptions.ClientError:
        sm.create_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=CONFIG_NAME,
            Tags=[{"Key": "Project", "Value": PROJECT}]
        )
        logger.info(f"Creating endpoint: {ENDPOINT_NAME}")

    # Wait for InService
    logger.info("Waiting for endpoint to reach InService (8-15 minutes)...")
    waiter = sm.get_waiter('endpoint_in_service')
    waiter.wait(
        EndpointName=ENDPOINT_NAME,
        WaiterConfig={'Delay': 30, 'MaxAttempts': 40}
    )
    logger.info(f"Endpoint is InService: {ENDPOINT_NAME}")


def run_smoke_tests() -> None:
    """Run three test invocations to verify the endpoint is working."""
    test_cases = [
        # (label, csv_input, expected_direction)
        ("High-risk patient",  "2.18,0.80,0.15,0.31,-0.54,0.17,0.47,0.52",  "> 0.5"),
        ("Low-risk patient",   "-0.84,-1.08,-0.54,-0.59,-0.54,-0.65,-0.90,-1.03", "< 0.5"),
        ("Borderline patient", "0.17,0.09,-0.28,-0.20,-0.43,-0.12,-0.01,0.15",   "~0.3–0.7"),
    ]

    logger.info("\n" + "=" * 50)
    logger.info("SMOKE TESTS")
    logger.info("=" * 50)
    for name, payload, expected in test_cases:
        resp  = sm_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='text/csv',
            Body=payload
        )
        score = float(resp['Body'].read().decode('utf-8').strip())
        level = "HIGH" if score >= 0.7 else ("MEDIUM" if score >= 0.3 else "LOW")
        logger.info(f"  [{name}]")
        logger.info(f"    Score: {score:.4f} (expected {expected})  →  {level}")
    logger.info("=" * 50)


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("DEPLOYING SAGEMAKER ENDPOINT — HEALTHPREDICT AI")
    logger.info("=" * 60)

    arn = get_latest_approved_model_arn()
    create_model(arn)
    create_endpoint_config()
    deploy_endpoint()
    run_smoke_tests()

    logger.info("=" * 60)
    logger.info(f"Endpoint deployed: {ENDPOINT_NAME}")
    logger.info("IMPORTANT: Delete the endpoint after testing to avoid unnecessary charges.")
    logger.info("=" * 60)
