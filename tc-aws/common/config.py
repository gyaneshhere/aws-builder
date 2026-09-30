"""Shared settings. Values come from environment variables (or a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

os.environ.setdefault("SAGEMAKER_SUPPRESS_V2_WARNING", "1")

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


@dataclass(frozen=True)
class Settings:
    region: str = field(default_factory=lambda: os.environ.get("AWS_REGION", "us-east-1"))
    role_arn: str = field(default_factory=lambda: os.environ.get("SM_ROLE_ARN", ""))
    bucket: str = field(default_factory=lambda: os.environ.get("BUCKET", ""))      # blank = SageMaker default bucket
    prefix: str = field(default_factory=lambda: os.environ.get("PREFIX", "text-classification"))
    bedrock_model: str = field(default_factory=lambda: os.environ.get(
        "BEDROCK_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0"))

    def s3(self, *parts: str) -> str:
        return "s3://" + "/".join([self.resolved_bucket(), self.prefix, *parts])

    def resolved_bucket(self) -> str:
        if self.bucket:
            return self.bucket
        import sagemaker

        return sagemaker.Session(boto_session=self.boto()).default_bucket()

    def boto(self):
        import boto3

        return boto3.Session(profile_name=os.environ.get("AWS_PROFILE") or None, region_name=self.region)

    def sm_session(self):
        import sagemaker

        return sagemaker.Session(boto_session=self.boto(), default_bucket=self.bucket or None)

    def role(self) -> str:
        if self.role_arn:
            return self.role_arn
        import sagemaker

        return sagemaker.get_execution_role(sagemaker_session=self.sm_session())   # works inside SageMaker Studio


SETTINGS = Settings()
