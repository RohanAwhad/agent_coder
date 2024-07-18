import argparse
import dataclasses
import difflib
import openai
import os
import re
import subprocess
import yaml

from typing import Optional

from src import utils

argparser = argparse.ArgumentParser()
argparser.add_argument("project_name", type=str, help="The name of the project.")
args = argparser.parse_args()

# ===
# Memory
# ===
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
    
# ===
# Agents
# ===
@dataclasses.dataclass
class Agent:
  model: str
  name: str
  sys_prompt: str
  max_tokens: int
  history: History
  temperature: float = 0.8


# ===
# Project
# ===
CONDA_PREFIX = "/Users/rohan/miniconda3/envs/"

def get_diff(text1: str, text2: str) -> str:
  d = difflib.Differ()
  diff = d.compare(text1.splitlines(keepends=True), text2.splitlines(keepends=True))
  return ''.join(diff)

class Project:
  def __init__(self, project_name: str, workspace: str):
    """
    # setup project directory
    # - project_name/
    #   - main.py
    #   - tests/
    """
    self.project_path = f"{workspace}/{project_name}"
    os.makedirs(self.project_path, exist_ok=True)
    if os.path.exists(f"{self.project_path}/environment.yml"):
      self.conda_env_name = yaml.safe_load(utils.load_text(f"{self.project_path}/environment.yml"))['name']
    # self.tests_path = f"{self.project_path}/tests"

  # def write_code(self, script: str, filename: str = "main.py", is_test: bool = False) -> None:
  #   fn = f"{self.tests_path}/{filename}" if is_test else f"{self.project_path}/{filename}"
  #   utils.save_text(script, fn)

  # def read_code(self, filename: str, is_test: bool = False) -> str:
  #   fn = f"{self.tests_path}/{filename}" if is_test else f"{self.project_path}/{filename}"
  #   return utils.load_text(fn)

  def run_tests(self) -> str:
    # Run tests
    return "Tests passed successfully"


  def _update_conda_env_file(self, env_name: str, channels: list[str], dependencies: list[str]) -> None:
    env = {
      "name": env_name,
      "channels": channels,
      "dependencies": dependencies,
    }
    self.conda_env_name = env_name
    text = yaml.dump(env)
    utils.save_text(text, f"{self.project_path}/environment.yml")

  def create_conda_env(self, env_name: str, channels: list[str], dependencies: list[str]) -> None:
    self._update_conda_env_file(env_name, channels, dependencies)
    subprocess.run(f"conda env create -f {self.project_path}/environment.yml", shell=True, check=True)

  def update_conda_env(self, env_name: str, channels: list[str], dependencies: list[str]) -> str:
    curr_env = utils.load_text(f"{self.project_path}/environment.yml")
    self._update_conda_env_file(env_name, channels, dependencies)
    new_env = utils.load_text(f"{self.project_path}/environment.yml")
    subprocess.run(f"conda env update -f {self.project_path}/environment.yml", shell=True, check=True)
    return get_diff(curr_env, new_env)

  def create_folder(self, folderpath: str) -> None:
    os.makedirs(f"{self.project_path}/{folderpath}", exist_ok=True)

  def create_file(self, filepath: str) -> None:
    utils.save_text("", f"{self.project_path}/{filepath}")

  def read_file(self, filepath: str) -> str:
    ret = utils.load_text(f"{self.project_path}/{filepath}")
    new_ret = []
    for i, x in enumerate(ret.split("\n")):
      new_ret.append(f"{i+1}. {x}")
    return "\n".join(new_ret)


  def append_to_file(self, filepath: str, content: str, line_number: Optional[int] = None) -> str:
    curr_text = utils.load_text(f"{self.project_path}/{filepath}")
    if not line_number:
      utils.append_text(content, f"{self.project_path}/{filepath}")
      new_text = utils.load_text(f"{self.project_path}/{filepath}")
      return get_diff(curr_text, new_text)

    curr_text_lines = curr_text.split("\n")
    curr_text_lines.insert(int(line_number-1), content)
    new_text = "\n".join(curr_text_lines)
    utils.save_text(new_text, f"{self.project_path}/{filepath}")

    # calculate diff and return
    return get_diff(curr_text, new_text)

  def rewrite_portion(self, filepath: str, content: str, start_line: int, end_line: int) -> str:
    curr_text = utils.load_text(f"{self.project_path}/{filepath}")
    curr_text_lines = curr_text.split("\n")
    curr_text_lines = curr_text_lines[:start_line-1] + content.split("\n") + curr_text_lines[end_line:]
    new_text = "\n".join(curr_text_lines)
    utils.save_text(new_text, f"{self.project_path}/{filepath}")
    return get_diff(curr_text, new_text)

  def list_files(self, folderpath: str) -> str:
    files = os.listdir(f"{self.project_path}/{folderpath}")
    return "\n".join(files)

  def run_commands(self, commands: list[str]) -> str:
    process = subprocess.Popen(['bash'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    commands = [f"cd {self.project_path}"] + commands

    for cmd in commands:
      process.stdin.write(f"{cmd}\n")
      process.stdin.flush()
    
    output, error = process.communicate()
    process.terminate()

    ret = f"Output: {output}\nError: {error}"
    return ret
  
  def run_tests(self, filepath: str) -> str:
    commands = [
      f"{CONDA_PREFIX}/{self.conda_env_name}/bin/python -m pytest {filepath}",
    ]
    return self.run_commands(commands)

  def run_python(self, filepath) -> str:
    commands = [f"{CONDA_PREFIX}/{self.conda_env_name}/bin/python {filepath}"]
    return self.run_commands(commands)

# ===
# LLM
# ===
client = openai.OpenAI(
  api_key=os.environ.get("TOGETHER_API_KEY"),
  base_url="https://api.together.xyz/v1",
)




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
  return res


# ===
# Tools
# ===

def create_employee(job_role: str, employee_name: str):
  if job_role == "requirement_engineer":
    ret = Agent(
      model="Qwen/Qwen2-72B-Instruct",
      name=employee_name,
      sys_prompt=utils.load_text("prompts/requirement_engineer.txt"),
      max_tokens=2048,
      temperature=0.8,
      history=History(),
    )
  elif job_role == "test_designer":
    ret = Agent(
      model="deepseek-ai/deepseek-coder-33b-instruct",
      name=employee_name,
      sys_prompt=utils.load_text("prompts/test_designer_agent.txt"),
      max_tokens=2048,
      temperature=0.8,
      history=History(),
    )
  elif job_role == "programmer":
    ret = Agent(
      model="deepseek-ai/deepseek-coder-33b-instruct",
      name=employee_name,
      sys_prompt=utils.load_text("prompts/programmer_agent.txt"),
      max_tokens=2048,
      temperature=0.8,
      history=History(),
    )
  elif job_role == "environment_designer":
    ret = Agent(
      model="Qwen/Qwen2-72B-Instruct",
      name=employee_name,
      sys_prompt=utils.load_text("prompts/environment_designer_agent.txt"),
      max_tokens=2048,
      temperature=0.8,
      history=History(),
    )
  else:
    raise ValueError(f"Invalid job role: {job_role}")

  return ret, f"Employee {employee_name} created successfully."


if __name__ == '__main__':
  brain = Agent(
    model="Qwen/Qwen2-72B-Instruct",
    name="Brain",
    sys_prompt=utils.load_text("prompts/brain.txt"),
    max_tokens=2048,
    temperature=0.8,
    history=History(),
  )
  objective = input("What do you want to achieve today? ")
  prompt = objective
  employees = {}
  project = Project(args.project_name, './ai_workspace')
  while True:
    try:
      print('Prompt:', prompt)
      res = generate_response(brain, prompt, brain.history)

      brain.history.add("user", prompt)
      brain.history.add("assistant", res)

      print('Response:', res)


      yaml_ptrn = re.compile(r"<yaml>(.*?)</yaml>", re.DOTALL)
      match = yaml_ptrn.search(res)
      yaml_data = None
      if not match:
        raise ValueError("Couldn't find any YAML in the response. Please try again with <yaml> tags.")
      yaml_str = yaml_ptrn.search(res).group(1).strip()
      yaml_data: dict = yaml.safe_load(yaml_str)

      tool_uid = yaml_data["tool_uid"]
      args = yaml_data['args']
      for k, v in args.items():
        if isinstance(v, str):
          args[k] = v.strip()
      
      if tool_uid == "create_employee":
        employee, prompt = create_employee(**args)
        employees[employee.name] = employee
      elif tool_uid == "chat":
        employee_name = args['employee_name']
        employee = employees[employee_name]
        msg_to_employee = args['msg']
        emp_res = generate_response(employee, msg_to_employee, employee.history)
        employee.history.add("user", msg_to_employee)
        employee.history.add("assistant", emp_res)
        prompt = emp_res
      elif tool_uid == "run_tests":
        prompt = project.run_tests(**args)
      elif tool_uid == "create_conda_env":
        project.create_conda_env(**args)
        prompt = "Conda environment created successfully"
      elif tool_uid == "update_conda_env":
        project.update_conda_env(**args)
        prompt = "Conda environment updated successfully"
      elif tool_uid == "create_folder":
        project.create_folder(**args)
        prompt = "Folder created successfully"
      elif tool_uid == "create_file":
        project.create_file(**args)
        prompt = "File created successfully"
      elif tool_uid == "read_file":
        prompt = project.read_file(**args)

      elif tool_uid == "append_to_file":
        content_ptrn = re.compile(r"<content>(.*?)</content>", re.DOTALL)
        match = content_ptrn.search(res)
        if not match:
          raise ValueError("Couldn't find anything in <content> tags in the response. Please try again.")
        content = match.group(1)
        prompt = project.append_to_file(content=content, **args)
        prompt = "Content appended successfully. Below is the diff\n" + prompt

      elif tool_uid == "rewrite_portion":
        content_ptrn = re.compile(r"<content>(.*?)</content>", re.DOTALL)
        match = content_ptrn.search(res)
        if not match:
          raise ValueError("Couldn't find anything in <content> tags in the response. Please try again.")
        content = match.group(1)
        prompt = project.rewrite_portion(content=content, **args)
        prompt = "Content rewritten successfully. Below is the diff\n" + prompt

      elif tool_uid == "list_files":
        prompt = project.list_files(**args)

      elif tool_uid == "ask_user_input":
        print(args['message'])
        prompt = input("Enter your response: ")

      elif tool_uid == "shutdown":
        exit_code = args['exit_code']
        if exit_code != 0:
          print('Encountered an error.')
          print(args['error_message'])
        
        print("Shutting down...")
        break
      elif tool_uid == "run_commands":
        prompt = project.run_commands(**args)
      elif tool_uid == "run_python_file":
        prompt = project.run_python(**args)
      else:
        raise ValueError("Invalid tool UID.")

    except Exception as e:
      prompt = f"Error: {e}"
      print(prompt)
      suffix = input("Do you want to help? (Enter to continue anyways): ")
      prompt += suffix