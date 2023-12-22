FROM python:3.10
RUN echo 'Build is launched'

WORKDIR /Users/antonivanov/PycharmProjects/ml_ops

COPY .. 
ADD ..

RUN python -m pip install --no-cache-dir poetry==1.4.2 \
    && poetry config virtualenvs.create false \
    && poetry install --without dev,test --no-interaction --no-ansi \
    && rm -rf $(poetry config cache-dir)/{cache,artifacts}


COPY /Users/antonivanov/PycharmProjects/ml_ops
CMD poetry run python main.py