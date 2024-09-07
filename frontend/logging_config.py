import logging
import os

# Create logs directory if not exists
if not os.path.exists('logs'):
    os.makedirs('logs')

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Handlers for each module's log file
streamlit_handler = logging.FileHandler("logs/streamlit.log")
streamlit_handler.setLevel(logging.DEBUG)

server_handler = logging.FileHandler("logs/server.log")
server_handler.setLevel(logging.DEBUG)

pipeline_handler = logging.FileHandler("logs/pipeline.log")
pipeline_handler.setLevel(logging.DEBUG)

# Define logger for each part of the application
def get_logger(name):
    logger = logging.getLogger(name)
    if "streamlit" in name:
        logger.addHandler(streamlit_handler)
    elif "server" in name:
        logger.addHandler(server_handler)
    elif "pipeline" in name:
        logger.addHandler(pipeline_handler)
    return logger
