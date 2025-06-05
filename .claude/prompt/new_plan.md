Z:\taf\chat_ui\deepseek-engineer\deepseek-eng.py -- one thing I want from this repo is the read tools {
          "type": "function",
          "function": {
              "name": "read_file",
              "description": "Read the content of a single file from the filesystem",
              "parameters": {
                  "type": "object",
                  "properties": {
                      "file_path": {
                          "type": "string",
                          "description": "The path to the file to read (relative or absolute)",
                      }
                  },
                  "required": ["file_path"]
              },
          }
      },
      {
          "type": "function",
          "function": {
              "name": "read_multiple_files",
              "description": "Read the content of multiple files from the filesystem",
              "parameters": {
                  "type": "object",
                  "properties": {
                      "file_paths": {
                          "type": "array",
                          "items": {"type": "string"},
                          "description": "Array of file paths to read (relative or absolute)",
                      }
                  },
                  "required": ["file_paths"]
              },
          }
      },
      
this would allow me to close the loop. More capable AI's can directly use the tools no need to output, they can read and analyse the scripts with the user (now doing co-analyse   ││   through discussion), less capable models can generate output.. then reload with fresh prompt (geared towards co-analysis)  and tools (read tools) and then do the co-analysis.