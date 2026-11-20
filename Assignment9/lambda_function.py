import json, os, boto3

runtime = boto3.client("sagemaker-runtime")
ENDPOINT = os.environ["ENDPOINT_NAME"]

def lambda_handler(event, context):
    try:
        features = json.loads(event["body"])["features"]
        if not isinstance(features, list) or len(features) != 30:
            return {"statusCode": 400,
                    "body": json.dumps({"error": "Expected 30 features"})}

        resp = runtime.invoke_endpoint(
            EndpointName=ENDPOINT,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps([[float(x) for x in features]]),
        )
        raw = resp["Body"].read()
        print("RAW RESPONSE:", raw)
        label = int(json.loads(raw)[0])

        return {"statusCode": 200,
                "body": json.dumps({"prediction": "benign" if label == 1 else "malignant"})}
    except (KeyError, ValueError, TypeError):
        return {"statusCode": 400,
                "body": json.dumps({"error": "Invalid input"})}
    except Exception as e:
        print("ERROR:", repr(e))
        return {"statusCode": 500,
                "body": json.dumps({"error": "Internal error"})}