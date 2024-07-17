# Coder Agent
Build an agent that at leasts gives a significant v0 of the code.

For the time being, there are 3 agents:
1. Requirements Engineer
2. Programmer
3. Test Designer

The requirements will be given in a chat interface to the requirements engineer. Once the human approval is given to the requirements, the requirements engineer will pass the requirements to the programmer and tester.

The programmer will write out the functions that are being asked for in a single file. Simultanously, the tester will write out the test cases for the functions that are being asked for in a single file.

### ToDos:
- [ ] The prompt for requirement engineer should take into account all the necessary information that is given. Eg. model names, frameworks to use, etc.
- [ ] Test Executor Agent
- [ ] Being able to write env.yaml for conda environments


# Brain Dump
Coding Train Lecture on Autonomous agents for simulation: Nautre of Code
1. Desired Speed
2. Target
3. max Force
4. Steering behaviour
5. arriving behaviour
6. Field of view 
7. Scalar Projection
8. Path following using projection
  1. Check vehicles future location
  2. Project it onto desired path
  3. Move the projection a little bit further along the path. This will be the target. 
  4. Seek the target
9. Group Steering
  1. Alignment with neighbours
  2. Neighbors within a field
10. Flee & Separarion
11. Cohesion
12. Keep view clear


For Coding the flow could be:

# Single Solution Generation

1. Chat / Text file.
2. Requirements Generation
  1. Will generate project name
  2. Will generate functions and their requirements
3. Generating tests
4. Using those tests and requirements generating code
5. Writing the test and code to the appropriate files 
  1. Creating files and folder structure
6. Executing the tests
7. If the code passes all of them, go to 12. Else, continue.
8. Generate reason for why it didn’t pass, and generate a fix (same call)
9. Edit the code file in the appropriate location.
10. Go to step 7
11. If more tasks: Go to the next task/function in line and then go to step 4.
12. <END>

## Agents:
1. Requirement Engineer
2. Programmer -> Generate Code
3. Test Designer
4. Test Executor
5. Writer
  1. Given code and text write to a file
  2. Create files & folders
  3. Read files
  4. Search files
6. Reflection Agent
  1. Web Search
  2. File Search
  3. Read file

# Multiple Solutions Generation

1. Chat / Text file.
2. Requirements Generation
3. Generating plans for tests (Multiple)
  1. Select the ones that are good
  2. Generate codes for that test
  3. Select ones that are good
  4. Merge them in 1.
4. Using those tests and requirements, generate code (Multiple)
  1. Generate Multiple plans
  2. Select the top-k
  3. Generate multiple codes
  4. Select the top-k
5. Writing the test and code to the appropriate files (How am I going to execute multiple codes?)
  1. Creating files and folder structure
6. Executing the tests
7. If the code passes all of them, go to 12. Else, continue.
8. Generate reason for why it didn’t pass, and generate a fix (same call)
9. Edit the code/test file in the appropriate location.
10. Go to step 7
11. If more tasks: Go to the next task/function in line and then go to step 4.
12. <END>

# Agents:
1. Requirement Engineer
2. Programmer -> Generate Code
3. Test Designer
4. Test Executor
5. Writer
  1. Given code and text write to a file
  2. Create files & folders
6. Reflection Agent
