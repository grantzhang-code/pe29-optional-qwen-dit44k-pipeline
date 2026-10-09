"""Google Cloud Storage access from Canva AWS pods through Workload Identity Federation."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse


SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"invalid GCS URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def get_google_wif_credentials() -> tuple[Any, str | None]:
    """Exchange the current pod's AWS credentials for explicit Google credentials."""

    import boto3.session
    import google.auth.aws as google_auth_aws

    class AwsCredentialsSupplier(google_auth_aws.AwsSecurityCredentialsSupplier):
        def __init__(self, session: Any) -> None:
            self._session = session

        def get_aws_security_credentials(self, context: Any, request: Any) -> Any:
            del context, request
            credentials = self._session.get_credentials()
            if credentials is None:
                raise RuntimeError("No AWS credentials available for Google WIF")
            frozen = credentials.get_frozen_credentials()
            return google_auth_aws.AwsSecurityCredentials(
                access_key_id=frozen.access_key,
                secret_access_key=frozen.secret_key,
                session_token=frozen.token,
            )

        def get_aws_region(self, context: Any, request: Any) -> str:
            del context, request
            return self._session.region_name or os.environ.get("AWS_REGION", "us-east-1")

    session = boto3.session.Session(region_name=os.environ.get("AWS_REGION", "us-east-1"))
    service_account = os.environ["GOOGLE_WIF_SERVICE_ACCOUNT"]
    credentials = google_auth_aws.Credentials(
        audience=os.environ["GOOGLE_WIF_AUDIENCE"],
        subject_token_type="urn:ietf:params:aws:token-type:aws4_request",
        token_url="https://sts.googleapis.com/v1/token",
        service_account_impersonation_url=(
            "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/"
            f"{service_account}:generateAccessToken"
        ),
        aws_security_credentials_supplier=AwsCredentialsSupplier(session),
        scopes=SCOPES,
    )
    return credentials, os.environ.get("GOOGLE_CLOUD_PROJECT")


def make_gcs_client() -> Any:
    from google.cloud import storage

    credentials, project = get_google_wif_credentials()
    return storage.Client(project=project, credentials=credentials)


def download_gcs_bytes(uri: str, *, client: Any | None = None) -> bytes:
    bucket, key = parse_gcs_uri(uri)
    client = client or make_gcs_client()
    return client.bucket(bucket).blob(key).download_as_bytes()


def download_gcs_file(uri: str, destination: str, *, client: Any | None = None) -> None:
    bucket, key = parse_gcs_uri(uri)
    client = client or make_gcs_client()
    client.bucket(bucket).blob(key).download_to_filename(destination)
