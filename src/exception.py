import sys

def exception_message_details(error, error_detail:sys):
  """
  Generates a detailed error message for a given exception.

  Parameters:
    error (Exception): The exception object.
    error_detail (module): The sys module.

  Returns:
    str: The formatted error message containing the name of the Python script, the line number, and the error message.
  """
  _, _, exc_tb = error_detail.exc_info()
  file_name = exc_tb.tb_frame.f_code.co_filename
  error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
    file_name, exc_tb.tb_lineno, str(error)
  )

  return error_message

class CustomException(Exception):
  """
    Initializes the CustomException object.

    Parameters:
        error_message (any): The error message for the exception.
        error_detail (module): The sys module.

    Returns:
        None
  """
  def __init__(self, error_message, error_detail:sys):
    super().__init__(error_message)
    self.error_message = exception_message_details(error_message, error_detail=error_detail)

  def __str__(self):
    return self.error_message
  
if __name__ == "__main__":
  try:
    a = 1/0
  except Exception as e:
    raise CustomException(e, sys)