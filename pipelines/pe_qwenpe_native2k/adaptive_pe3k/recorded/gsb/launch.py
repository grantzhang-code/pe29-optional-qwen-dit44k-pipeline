"""Launch scoring with explicit workload WIF inside the existing UTP holder."""
import os
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent
os.environ.update(CANVA_FLAVOR='prod', FLAVOR='prod', CANVA_PLATFORM='eks',
    AWS_REGION='us-east-1', AWS_DEFAULT_REGION='us-east-1',
    GOOGLE_CLOUD_PROJECT='cnvprd-core-cn-npd', GOOGLE_CLOUD_LOCATION='global', GOOGLE_GENAI_USE_VERTEXAI='True',
    GOOGLE_WIF_AUDIENCE='//iam.googleapis.com/projects/653103247366/locations/global/workloadIdentityPools/vertex-wif/providers/wif-provider',
    GOOGLE_WIF_SERVICE_ACCOUNT='vertex@cnvprd-core-cn-npd.iam.gserviceaccount.com',
    CANVA_CLUSTER_STORAGE_PATH=str(ROOT/'durable'))
runpy.run_path(str(ROOT/'driver.py'),run_name='__main__')
