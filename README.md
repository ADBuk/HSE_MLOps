# HSE_MLOps
hw for HSE MLOps course

### Файлы
1. `main.py` : попытка создать фаст апи
2. `models_handler.py`: файл с классом, который, в идеале, должен выполнять все функции из первой ДЗ.
3. `test.py` : тесты для апи (работает только вызов списка моделей, доступных на обучение)
4. `pyproject.toml` : проект поэтри
5. `poetry.lock` : зависимости через поэтри

### Как использовать
Установка всех библиотек
```
poetry install
```
Запуск самой API
```
poetry run main.py
```
В идеале, запустить тесты (TO BE DONE)
```
poetry run test.py
```
### HW 2
исправил ошибки в первой домашке, теперь все работает

С DVC не разобрался, что надо сделать было, потыкал в терминале

добавил работу в Minio + Docker

Запуск:
```
docker compose up --abort-on-container-exit
```

blessrng не хватает 2 балла до 4)))) Пощады, готов на жоски gachi-штраф
