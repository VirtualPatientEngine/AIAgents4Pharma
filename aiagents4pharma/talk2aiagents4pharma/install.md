###### Notes for Windows Users

If you are using Windows, it is recommended to install **Git Bash** for a smoother experience when running the bash commands in this guide.

- For applications that use **Docker Compose**, Git Bash is **required**.
- For applications that use **docker run** manually, Git Bash is **optional**, but recommended for consistency.

You can download Git Bash here: [Git for Windows](https://git-scm.com/downloads).

When using Docker on Windows, make sure you **run Docker with administrative privileges** if you face permission issues.

To resolve for permission issues, you can:

- Review the official Docker documentation on [Windows permission requirements](https://docs.docker.com/desktop/setup/install/windows-permission-requirements/).
- Alternatively, follow the community discussion and solutions on [Docker Community Forums](https://forums.docker.com/t/error-when-trying-to-run-windows-containers-docker-client-must-be-run-with-elevated-privileges/136619).

**LangSmith** support is optional. To enable it, create an API key [here](https://docs.smith.langchain.com/administration/how_to_guides/organization_management/create_account_api_key).

###### About startup.sh

Run the startup script. It will:

- Detect your hardware configuration (NVIDIA GPU, AMD GPU, or CPU). Apple Metal is unavailable inside Docker, and Intel SIMD optimizations are automatically handled without special configuration.
- Choose the correct Ollama image (`latest` or `rocm`)
- Launch the Ollama container with appropriate runtime settings
- Pull the required embedding model (`nomic-embed-text`)
- Start the agent **after the model is available**

###### To run multiple agents simultaneously

- Be sure to **replace the placeholder values** with your actual credentials before running any container:

  - `<your_openai_api_key>`
  - `<your_nvidia_api_key>`
  - `<your_zotero_api_key>`
  - `<your_zotero_user_id>`

- All agents default to **port `8501`**. If you plan to run multiple agents simultaneously, make sure to assign **different ports** to avoid conflicts.

  Example (Talk2Scholars on port `8502`):

  ```docker
  docker run -d \
    --name talk2scholars \
    -e OPENAI_API_KEY=<your_openai_api_key> \
    -e ZOTERO_API_KEY=<your_zotero_api_key> \
    -e ZOTERO_USER_ID=<your_zotero_user_id> \
    -e NVIDIA_API_KEY=<your_nvidia_api_key> \
    -p 8502:8501 \
    virtualpatientengine/talk2scholars
  ```

  Then access the app at: [http://localhost:8502](http://localhost:8502)
