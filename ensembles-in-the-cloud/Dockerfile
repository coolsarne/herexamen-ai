FROM europe-docker.pkg.dev/vertex-ai/training/tf-cpu.2-4:latest

WORKDIR /

# Copies the trainer code to the docker image.
COPY trainer /trainer

RUN pip install -r trainer/requirements.txt

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.train_model"]