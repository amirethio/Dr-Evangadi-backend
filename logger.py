import boto3
import uuid
import logging
from datetime import datetime
from botocore.exceptions import BotoCoreError, ClientError

dynamodb = boto3.resource("dynamodb", region_name="us-east-1")  # Adjust region if needed
table_name = "DrEvangadiResults"  # Your table name

def log_diabetes_prediction(input_data: dict, prediction: int, probability: float):
    try:
        logging.info("Attempting to log prediction to DynamoDB")
        table = dynamodb.Table(table_name)
        item = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "prediction_type": "diabetes",
            "input_data": input_data,
            "label": prediction,
            "probability": probability
        }
        response = table.put_item(Item=item)
        logging.info(f"DynamoDB put_item response: {response}")
        logging.info(f"Logged diabetes prediction with id: {item['id']}")
    except (BotoCoreError, ClientError) as e:
        logging.error(f"Error writing to DynamoDB: {e}")
    except Exception as e:
        logging.error(f"Unexpected error in DynamoDB logging: {e}")
