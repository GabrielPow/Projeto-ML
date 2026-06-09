from google.cloud import bigquery
import subprocess
import pandas as pd

ID_PROJECT = 'groova-pml-20261'
LOCATION = 'southamerica-east1'

def client_bigquery():
    subprocess.run(['gcloud','auth','application-default','login'], check=True)

    client = bigquery.Client(project=ID_PROJECT,
                            location=LOCATION)
    return client

def get_dfFeatures():
    client = client_bigquery()
    query = """
        SELECT
            f.*,
            r.nota,
            r.valor_oferta
        FROM `{ID_PROJECT}.groova_features.groova_features_1000` f
        LEFT JOIN `{ID_PROJECT}.groova_base_full.groova_dataset_raw` r
        ON f.artist_key = r.artist_key
    """
    df = client.query(query).to_dataframe()

    return df

