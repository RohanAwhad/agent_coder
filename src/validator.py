import yaml

from typing import Optional

def validate_requirements_yaml(yaml_content: str) -> tuple[bool, Optional[dict]]:
    """
    Validates the structure of a YAML string against a specified format for a project.

    Parameters:
    - yaml_content (str): The YAML content as a string.

    Returns:
    - bool: True if the YAML structure is valid, False otherwise.
    """
    # Define the expected keys and their types
    required_keys = {
        'project_name': str,
        'requirements': list
    }
    
    requirement_keys = {
        'task_description': str,
        'function_signature': str,
        'docstring': str
    }

    # Parse the YAML content
    try:
        data = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
        return False, None

    # Check top-level keys
    if not isinstance(data, dict) or any(key not in data for key in required_keys):
        print("Missing one or more required keys at the top level or invalid structure.")
        return False, None

    # Check top-level key types
    if not isinstance(data['project_name'], required_keys['project_name']):
        print("Invalid type for project_name.")
        return False, None
    
    # Check requirements list
    if not all(isinstance(req, dict) for req in data['requirements']):
        print("Each requirement must be a dictionary.")
        return False, None

    # Validate each requirement entry
    for req in data['requirements']:
        # Check for each required key in requirements
        if any(key not in req for key in requirement_keys):
            print("Missing one or more required keys in a requirement.")
            return False, None
        # Check types of each field in requirements
        if not all(isinstance(req[key], requirement_keys[key]) for key in requirement_keys):
            print(f"Invalid type for one or more keys in requirements: {req}")
            return False, None
    
    print("YAML structure is valid.")
    return True, data


import tempfile
import subprocess

from io import StringIO
from pylint.lint import Run
from pylint.reporters.text import TextReporter

def extract_errors(error_string):
  error_list = []
  lines = error_string.split('\n')
  
  for line in lines:
    if ': E' in line:
      fn, line_no, col_no, error_code, error_message = map(lambda x: x.strip(), line.split(':'))

      if error_code == "E0602":  # undefined-variable
        error_list.append(f"Error: {error_message} at line {line_no} column {col_no}")
  
  return error_list

def check_python_script_for_errors(script_str) -> list[str]:
  # Save to a temporary file
  with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp:
    temp_name = temp.name
    temp.write(script_str.encode())
    temp.close()

  # Run pylint on the temporary file, disabling all but error messages
  pylint_output = StringIO()  # Custom open stream
  reporter = TextReporter(pylint_output)
  Run(['-d=C,R,W', temp_name], reporter=reporter, exit=False)
  result = pylint_output.getvalue()

  # Cleanup the temporary file
  subprocess.run(["rm", temp_name])

  errors_list = extract_errors(result)
  return errors_list



if __name__ == "__main__":
    script = """# test_load_image.py

import pytest

import main

def test_load_image_valid():
    image = main.load_image('valid_image_path')
    assert type(image) == Image, "Image should be loaded successfully"

def test_load_image_invalid():
    with pytest.raises(Exception):
        image = main.load_image('invalid_image_path')

def test_load_image_not_exist():
    with pytest.raises(Exception):
        image = main.load_image('not_exist_image_path')

def test_load_image_not_image():
    with pytest.raises(Exception):
        image = main.load_image('not_image_path')
"""
    check_python_script_for_errors(script)