# AIO2024_EXERCISES _ Project RAG LLM with Chainlit interface
Repo is using python 3.11 version.

## Getting Started
This project has been tested using Python 3.11 version.

## Steps

1. **Install Virtual Environment**
    - Ensure `virtualenv` is installed. You can install it using the following commands:
      ```sh
      ~$ pip install virtualenv
      # or
      ~$ pip3 install virtualenv
      ```

2. **Clone the Repository**
    - Clone this repository using the following command:
      ```sh
      ~$ git clone https://github.com/Truongvo307/AIO2024_EXERCISES.git
      ```
      
3. **Navigate to the Project Directory**
    - Change to the cloned repository directory:
      ```sh
      ~$ cd <your_path>
      ```

4. **Create a Virtual Environment**
    - Use `virtualenv` to create a virtual environment for the project:
      ```sh
      ~$ virtualenv venv --python=python3.11
      ```
      #or 
      ~$ python3.11 -m venv venv  
      *where `venv-dfl` is the name given to your virtual environment.*

5. **Activate the Virtual Environment**
    - Activate your virtual environment with the following command:
      ```sh
      # On Linux OS
      ~$ source ./venv-dfl/bin/activate 
      # On Window OS
      ~$ ./venv/Script/activate
      ```

6. **Install Dependencies**
    - Install the necessary dependencies using the `requirements.txt` file:
      ```sh
      ~$ pip install -r ./requirements.txt
      ```

7. **Run Chainlit App**
    - Start the Chainlit app with the following command:
      ```sh
      ~$ chainlit run app.py --host 0.0.0.0 --port 8000 &>/content/logs.txt
      ```

8. **Launch Chainlit App with Localtunnel**
    - Use `localtunnel` to expose the app with a subdomain:
      ```sh
      ~$ lt --port 8000 --subdomain aivn-simple-rag
      ```
