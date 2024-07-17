import dataclasses
import openai
import os
import re
import yaml

from typing import Optional

from src import validator
from src import utils


# ===
# Agents
# ===
@dataclasses.dataclass
class Agent:
  model: str
  sys_prompt: str
  max_tokens: int
  temperature: float = 0.8


requirement_agent = Agent(
  model="Qwen/Qwen2-72B-Instruct",
  sys_prompt=utils.load_text("prompts/requirement_engineer.txt"),
  max_tokens=2048,
  temperature=0.8,
)
test_designer_agent = Agent(
  model="deepseek-ai/deepseek-coder-33b-instruct",
  sys_prompt=utils.load_text("prompts/test_designer_agent.txt"),
  max_tokens=2048,
  temperature=0.8,
)
programmer_agent = Agent(
  model="deepseek-ai/deepseek-coder-33b-instruct",
  sys_prompt=utils.load_text("prompts/programmer_agent.txt"),
  max_tokens=2048,
  temperature=0.8,
)

# ===
# Project Workspace
# ===

class Project:
  def __init__(self, project_name: str, workspace: str):
    """
    # setup project directory
    # - project_name/
    #   - main.py
    #   - tests/
    """
    self.project_path = f"{workspace}/{project_name}"
    self.tests_path = f"{self.project_path}/tests"

  def setup(self) -> None:
    os.makedirs(self.project_path, exist_ok=True)
    os.makedirs(self.tests_path, exist_ok=True)
    utils.save_text("", f"{self.project_path}/main.py")

  def write_code(self, script: str, filename: str = "main.py", is_test: bool = False) -> None:
    fn = f"{self.tests_path}/{filename}" if is_test else f"{self.project_path}/{filename}"
    utils.save_text(script, fn)

  def read_code(self, filename: str, is_test: bool = False) -> str:
    fn = f"{self.tests_path}/{filename}" if is_test else f"{self.project_path}/{filename}"
    return utils.load_text(fn)


# ===
# LLM
# ===
client = openai.OpenAI(
  api_key=os.environ.get("TOGETHER_API_KEY"),
  base_url="https://api.together.xyz/v1",
)

@dataclasses.dataclass
class Message:
  role: str
  content: str
  def to_dict(self) -> dict:
    return dataclasses.asdict(self)

class History:
  def __init__(self):
    self.messages: list[Message] = []

  def add(self, role: str, content: str) -> None:
    msg = Message(role=role, content=content)
    self.messages.append(msg)


def generate_response(agent: Agent, prompt: str, history: Optional[History] = None) -> str:
  messages = [{"role": "system", "content": agent.sys_prompt}]
  if history:
    for msg in history.messages: messages.append(msg.to_dict())
  messages.append({"role": "user", "content": prompt.strip()})

  response = client.chat.completions.create(
    model=agent.model,
    messages=messages,
    max_tokens=agent.max_tokens,
    temperature=agent.temperature,
  )
  res = response.choices[0].message.content.strip()
  # if not res.startswith('<planning>'): res = f"<planning>{res}"
  return res


# ===
# Required Functions
# ===
MAX_RETRIES = 3
def generate_requirements(job: str) -> tuple[Optional[dict], str]:
  global MAX_RETRIES, requirement_agent

  retries = MAX_RETRIES
  while retries > 0:
    job_reqs = generate_response(requirement_agent, job)
    pattern = re.compile(r"<yaml>(.*?)</yaml>", re.DOTALL)
    yaml_str = pattern.search(job_reqs).group(1).strip()
    is_valid, data = validator.validate_requirements_yaml(yaml_str)
    if is_valid: return data, job_reqs
    retries -= 1
    print('Retrying...')
  return None, job_reqs


def generate_tests(requirements: dict, project_loc: str) -> tuple[Optional[str], Optional[str], str]:
  print('Writing tests for:', requirements['task_description'])
  pattern = re.compile(r"<python>(.*?)</python>", re.DOTALL)
  tmp = dict(
    signature=reqs['function_signature'].strip(),
    docstring=reqs['docstring'].strip(),
  )
  retries = MAX_RETRIES
  prompt = yaml.dump(tmp)
  hist = History()
  while retries:
    tests = generate_response(test_designer_agent, prompt)
    hist.add("user", prompt)
    hist.add("assistant", tests)
    match = pattern.search(tests)
    if match:
        python_script = match.group(1).strip()
    else:
        python_script = ""
    if not python_script:
      retries -= 1
      print('Retrying...')
      continue

    filename = extract_filename_from_first_line(python_script)
    if not filename:
      retries -= 1
      print('Retrying...')
      continue

    return python_script, filename, tests

    # validate with pylint
    # err_list = validator.check_python_script_for_errors(python_script)
    # if len(err_list) == 0: return python_script, filename, tests

    prompt = f'After running the code through pylint, got the following errors:\n- {'\n- '.join(err_list)}\n\nPlease fix the errors and give the whole script.'
    retries -= 1
    print(prompt)
    print('Retrying...')

  return None, None, tests

def extract_filename_from_first_line(script_content: str) -> str:
  """
  Extracts the filename from the first line of the given script if it contains a filename comment.

  Parameters:
  - script_content (str): The content of the Python script.

  Returns:
  - str: The extracted filename or an empty string if no filename comment is found.
  """
  # Split the content into lines
  lines = script_content.strip().split('\n')
  
  if not lines:
    return ""
  
  # Define the regex pattern for extracting the filename
  pattern = re.compile(r'^# (\w+\.py)$')
  
  # Check the first line with the regex pattern
  first_line = lines[0].strip()
  match = pattern.search(first_line)
  
  if match:
    return match.group(1)
  else:
    return ""


def generate_code(reqs: dict, history: History, project: Project) -> tuple[Optional[str], str]:
  pattern = re.compile(r"<python>(.*?)</python>", re.DOTALL)
  test_fn = func_sign_to_test_file[reqs['function_signature']]
  test = project.read_code(test_fn, is_test=True)
  yaml_dict = dict(
    function_signature=reqs['function_signature'].strip(),
    task_description=reqs['task_description'].strip(),
  )
  prompt = f"<yaml>\n{yaml.dump(yaml_dict)}\n</yaml>\n<test>\n{test}\n</test>"

  print('Writing code for:', reqs['task_description'])
  retries = MAX_RETRIES
  while retries:
    code = generate_response(programmer_agent, prompt, history)
    match = pattern.search(code)
    python_script = match.group(1).strip() if match else ""
    if not python_script:
      retries -= 1
      print('Retrying...')
      continue

    history.add("user", prompt)
    history.add("assistant", code)
    return python_script, code

  return None, code


# ===
# Main
# ===
if __name__ == "__main__":
  # Variables for file paths
  AI_WORKSPACE = "./ai_workspace"
  SAVE_DIR = "./gen"

  os.makedirs(AI_WORKSPACE, exist_ok=True)
  os.makedirs(SAVE_DIR, exist_ok=True)

  TEST_JOB_FILE = f"test_job1.txt"
  JOB_REQS_FILE = f"{SAVE_DIR}/job1_requirements.txt"
  JOB_REQS_JSON_FILE = f"{SAVE_DIR}/job1_requirements.json"
  JOB_TESTS_FILE = f"{SAVE_DIR}/job_tests.txt"
  JOB_CODES_WITH_CONTEXT_FILE = f"{SAVE_DIR}/job1_codes_with_context_2.txt"
  JOB_CODES_FILE = f"{SAVE_DIR}/job1_codes.py"

  # Load and print job description
  job = utils.load_text(TEST_JOB_FILE)
  print('Job:', job)

  # # Generate requirements
  # req_dict, job_reqs = generate_requirements(job)
  # utils.save_json(req_dict, JOB_REQS_JSON_FILE)
  # utils.save_text(job_reqs, JOB_REQS_FILE)
  # if req_dict is None:
  #   print('Failed to generate requirements.')
  #   print('Check out the latest generated requirements at', JOB_REQS_FILE)
  #   exit()

  # Load requirements
  job_reqs = utils.load_text(JOB_REQS_FILE)
  print('Requirements:', job_reqs)
  req_dict = utils.load_json(JOB_REQS_JSON_FILE)
  
  # setup project directory
  project = Project(req_dict['project_name'], AI_WORKSPACE)
  project.setup()

  # Generate tests
  all_tests = []
  func_sign_to_test_file = {}
  for reqs in req_dict['requirements']:
    python_script, filename, tests = generate_tests(reqs, project.project_path)
    all_tests.append(tests)
    print(tests)

    if python_script:
      project.write_code(python_script, filename, is_test=True)
      func_sign_to_test_file[reqs['function_signature']] = filename
    else:
      print('Failed to generate tests.')
      print('Check out the latest generated tests at', JOB_TESTS_FILE)

  utils.save_text("\n---\n".join(all_tests), JOB_TESTS_FILE)

  # Generate code with history context
  all_codes = []
  history = History()
  for reqs in req_dict['requirements']:
    python_script, code = generate_code(reqs, history, project)
    if python_script:
      project.write_code(python_script+'\n\n')

    all_codes.append(code)
    print(code)
    print()

  utils.save_text("\n---\n".join(all_codes), JOB_CODES_WITH_CONTEXT_FILE)

  # Save final code file
  full_code = utils.load_text(JOB_CODES_WITH_CONTEXT_FILE)
  pattern = re.compile(r"<python>(.*?)</python>", re.DOTALL)
  utils.save_text("# This code is generated by AI\n\n", JOB_CODES_FILE)
  for split in full_code.split("---"):
      code = pattern.search(split).group(1).strip()
      utils.append_text(code + '\n\n', JOB_CODES_FILE)