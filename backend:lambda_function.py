import json
import os
import time
import boto3
from botocore.exceptions import ClientError


# ---------- Helpers ----------

def _response(status_code: int, body_obj: dict):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "content-type",
            "Access-Control-Allow-Methods": "OPTIONS,POST"
        },
        "body": json.dumps(body_obj)
    }


def _safe_json_loads(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None


def _extract_message(event: dict) -> str:
    """
    Supports:
    - Lambda test events: {"message":"hi"}
    - API Gateway proxy: {"body":"{\"message\":\"hi\"}"}
    - API Gateway v2: event.get("body") string
    """
    if not isinstance(event, dict):
        return ""

    # Direct
    if isinstance(event.get("message"), str):
        return event.get("message", "").strip()

    body = event.get("body")
    if body is None:
        return ""

    # Body can be dict or JSON string
    if isinstance(body, dict):
        msg = body.get("message", "")
        return (msg or "").strip()

    if isinstance(body, str):
        parsed = _safe_json_loads(body)
        if isinstance(parsed, dict):
            msg = parsed.get("message", "")
            return (msg or "").strip()

    return ""


def _load_kb_from_s3(bucket: str, key: str) -> dict:
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    raw = obj["Body"].read().decode("utf-8")
    kb = json.loads(raw)
    if not isinstance(kb, dict):
        return {}
    return kb


def _detect_sentiment_if_enabled(text: str) -> str | None:
    use_sentiment = os.environ.get("USE_SENTIMENT", "false").strip().lower() == "true"
    if not use_sentiment or not text:
        return None

    try:
        comprehend = boto3.client("comprehend")
        out = comprehend.detect_sentiment(Text=text, LanguageCode="en")
        return out.get("Sentiment")
    except Exception:
        # Do not fail the chatbot if Comprehend is unavailable
        return None


def _key_phrases(text: str) -> str:
    """
    Returns lowercased key phrases merged into a single string.
    Used to improve routing.
    """
    if not text:
        return ""

    try:
        comprehend = boto3.client("comprehend")
        out = comprehend.detect_key_phrases(Text=text, LanguageCode="en")
        phrases = " ".join([p.get("Text", "").lower() for p in out.get("KeyPhrases", []) if p.get("Text")])
        return phrases
    except Exception:
        return ""


def _lex_reply_if_configured(user_text: str) -> dict | None:
    """
    Optional: If Lex env vars exist, call Lex V2 runtime.
    Returns dict: {"reply": "...", "intent": "..."} or None if not configured / fails.
    """
    bot_id = os.environ.get("LEX_BOT_ID", "").strip()
    alias_id = os.environ.get("LEX_BOT_ALIAS_ID", "").strip()
    locale_id = os.environ.get("LEX_LOCALE_ID", "en_US").strip()

    if not bot_id or not alias_id or not user_text:
        return None

    try:
        lex = boto3.client("lexv2-runtime")
        session_id = f"web-{int(time.time())}"
        resp = lex.recognize_text(
            botId=bot_id,
            botAliasId=alias_id,
            localeId=locale_id,
            sessionId=session_id,
            text=user_text
        )

        intent_name = None
        interpretations = resp.get("interpretations", [])
        if interpretations and isinstance(interpretations, list):
            intent = interpretations[0].get("intent", {})
            intent_name = intent.get("name")

        messages = resp.get("messages", [])
        lex_text = None
        if messages and isinstance(messages, list):
            # pick first message content
            lex_text = messages[0].get("content")

        if lex_text:
            return {"reply": lex_text, "intent": intent_name or "LexIntent"}
        return None

    except Exception:
        return None


# ---------- Main Handler ----------

def lambda_handler(event, context):
    # Handle CORS preflight
    method = (event.get("requestContext", {}).get("http", {}).get("method")
              or event.get("httpMethod")
              or "").upper()
    if method == "OPTIONS":
        return _response(200, {"ok": True})

    user_msg = _extract_message(event)
    if not user_msg:
        return _response(400, {"reply": "Missing 'message' in request.", "intent": "BadRequest"})

    # Load KB
    kb_bucket = os.environ.get("KB_BUCKET", "").strip()
    kb_key = os.environ.get("KB_KEY", "knowledgebase.json").strip()

    if not kb_bucket:
        return _response(500, {"reply": "Server misconfigured: KB_BUCKET not set.", "intent": "ServerConfigError"})

    try:
        kb = _load_kb_from_s3(kb_bucket, kb_key)
    except ClientError as e:
        return _response(500, {"reply": f"KB load error from S3: {str(e)}", "intent": "KBLoadError"})
    except Exception as e:
        return _response(500, {"reply": f"KB load error: {str(e)}", "intent": "KBLoadError"})

    # 1) Try Lex (optional)
    lex_out = _lex_reply_if_configured(user_msg)
    if lex_out:
        sentiment = _detect_sentiment_if_enabled(user_msg)
        body = {
            "reply": lex_out["reply"],
            "intent": lex_out.get("intent") or "LexIntent"
        }
        if sentiment:
            body["sentiment"] = sentiment
        return _response(200, body)

    # 2) Otherwise do keyword + Comprehend phrases routing
    lower_msg = user_msg.strip().lower()
    phrases = _key_phrases(user_msg)
    text = f"{lower_msg} {phrases}".strip()

    # Greeting
    if any(g in lower_msg for g in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]):
        intent = "GreetingIntent"
        reply = kb.get(intent, "Hi! How can I help you today?")

    # Programs
    elif any(w in text for w in ["program", "course", "diploma", "degree", "tuition", "fee"]):
        intent = "ProgramInfoIntent"
        reply = kb.get(intent, "We offer multiple programs. Ask which field you’re interested in.")

    # Registration
    elif any(w in text for w in ["register", "registration", "enroll", "enrollment", "admission", "apply"]):
        intent = "RegistrationHelpIntent"
        reply = kb.get(intent, "You can register via the student portal. Need step-by-step help?")

    # Support
    elif any(w in text for w in ["support", "counsel", "advis", "career", "help", "guidance"]):
        intent = "SupportServicesIntent"
        reply = kb.get(intent, "We provide student support services such as advising and career help.")

    # Fallback
    else:
        intent = "FallbackIntent"
        reply = kb.get(intent, "I’m not sure about that yet. Ask about programs, registration, or support services.")

    sentiment = _detect_sentiment_if_enabled(user_msg)

    body = {
        "reply": reply,
        "intent": intent
    }
    if sentiment:
        body["sentiment"] = sentiment

    return _response(200, body)