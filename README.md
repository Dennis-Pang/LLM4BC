# MP_LLM4BC

### Main Directory Descriptions:

1. `Agent_pipeline_charts/`
   - Stores agent pipeline-related visualization files
   - Contains visual representations of various processing flows

2. `Visul_Expert_Knowledge/`
   - Stores visualization-related expert knowledge
   - Contains visualization rules and templates

3. `dbschema/`
   - Database schema definition files
   - Contains database table structures and relationship definitions

4. `results/`
   - Stores processing results for all cases
   - Each case directory contains:
     - `query_result.csv`: Query result data
     - `1_processed_data.csv`: Processed data
     - `generated_code.py`: Generated visualization code
     - `xxx.png`: Generated chart

### Main File Descriptions:

1. Agent Program Files:
   - `SQL_agent.py`: SQL query processing agent
   - `Visualization_agent.py`: Visualization processing agent
   - `Analysis_agent.py` and `Analysis_agent_2.py`: Data analysis agents

2. Configuration Files:
   - `requirements.txt`: Python package dependencies
   - `.env`: Environment variables configuration
   - `setup.py`: Project installation configuration

3. Utility Files:
   - `general_functions.py`: General utility functions

### Important Notes:
1. All agent programs depend on the utility functions in `general_functions.py`
2. Environment variables are configured in `.env` file
3. Dependencies in `requirements.txt` must be installed before running
4. Database schemas should be defined in the `dbschema/` directory
