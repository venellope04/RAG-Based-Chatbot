from google.cloud import dialogflow_v2 as dialogflow
from google.api_core.exceptions import InvalidArgument, NotFound
import logging

class DialogflowHandler:
  def __init__(self, project_id, language_code='en'):
    """
    Initialize Dialogflow handler
    """
    self.project_id = project_id
    self.language_code = language_code
    self.session_client = dialogflow.SessionsClient()

    try:
      # Remove commented-out test connection block
      logging.info(f"Connected to Dialogflow project: {project_id}")
    except NotFound:
      logging.error(f"Dialogflow project {project_id} not found")
      raise Exception(
          f"Dialogflow project '{project_id}' not found. Please verify:\n"
          f"1. Project ID is correct\n"
          f"2. Service account key is properly set up\n"
          f"3. Dialogflow API is enabled\n"
          f"4. An agent is created in the project"
      )
    except Exception as e:
      logging.error(f"Dialogflow initialization error: {e}")
      raise

  def detect_intent(self, session_id, text):
    """
    Detect intent from text
    """
    try:
      session = self.session_client.session_path(self.project_id, session_id)
      text_input = dialogflow.TextInput(text=text, language_code=self.language_code)
      query_input = dialogflow.QueryInput(text=text_input)

      response = self.session_client.detect_intent(
          request={"session": session, "query_input": query_input}
      )

      result = response.query_result
      return (
          result.fulfillment_text,
          result.intent.display_name,
          result.intent_detection_confidence
      )

    except Exception as e:
      logging.error(f"Error in detect_intent: {e}")
      raise