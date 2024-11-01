# Сервис Подсказок

## Описание

Этот проект реализует сервис подсказок для поиска товаров, позволяя быстро предлагать пользователю релевантные запросы на основе частичного ввода. Решение основано на реализации префиксного дерева.

## Формирование Подсказок

Для формирования подсказок используются несколько методов:

- `suggest_query`: основной метод, предоставляющий подсказки на основе введенного текста.
- `suggest_removed_char`: удаляет последний символ из запроса и предлагает варианты.
- `suggest_last_words`: генерирует предложения на основе последних слов запроса.
- `suggest_each_word`: создает подсказки для каждого слова в запросе независимо.

После применения этих методов формируется список подсказок, которые сортируются по популярности (весу) и отбираются в топ-k для предоставления пользователю.

## Запуск сервиса

Запуск сервиса:

```bash
bash run.sh
```

## Тестирование

Запуск всех тестов:

```bash
pytest .
```

Запуск тестов с подробным выводом:

```bash
pytest -svv .
```

Запуск отдельного теста:

```bash
pytest -svv tests/test_app.py::test_suggestions_recall
```

## Анализ кода с помощью Pylint

Запуск Pylint:

```bash
pylint app.py
```