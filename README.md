# Чек-ап вагона / Прогнозирование отправления вагонов в ремонт

## Solution
Repo contains 3 folders
1. background - about preliminary experiments without target data, it can be used for further experiments
2. eda - analytics
3. models - our solutions and experiments

## Task description
<aside>
ℹ️ **Задача:**
Создать модель прогнозирования даты отправления вагона в плановый ремонт
_________
1. спрогнозировать, что вагон отправится в ПР в течение месяца
2. спрогнозировать, что вагон отправится в ПР в течение 10 дней

</aside>

### 🤔 Описание проблемы

Отправка вагона в плановый ремонт может происходить по разным причинам, как по регламенту (срок/пробег), так и из-за накопления мелких дефектов:

- Было много текущих ремонтов;
- Не было вариантов на погрузку и т.д.

Этих причин много, и все они влияют на возможность осуществления ремонта.

Вагон отправляют в ремонт после получения уведомления об их неисправности. К сожалению, текущий процесс не позволяет распределять нагрузку на ремонтные депо, управлять последней заявкой на погрузку перед ремонтом и многое другое. Ваша задача – исправить это.

---

Бизнес ценность решения определяется числом скрытых взаимосвязей, которые были выявлены в данных. Критерии следующие - выявленная взаимосвязь описывает группу не меньше 100 вагонов, отправленных в плановый ремонт, и использует нетривиальные (т.е. не только остаточный пробег или факт ремонта за прошедший период) параметры. Таким образом, чем больше источников данных будет использовано и чем больше будет выявлено правил, по которым вагоны отправляют в плановый ремонт, тем выше бизнес-ценность вашего решения.

---

### ❌ Ограничения на технологии

- Язык: Python
- Использовать библиотеки с открытым доступом

### 💡 Данные

[Трек 2 / Чек-ап вагона - Google Drive](https://drive.google.com/drive/folders/1Qh5ZMSDOJ0d0XuInbW4DyiuhaZ1mfmOv?usp=drive_link)

1. **train_1.rar** - данные с 1 августа по 31 января
    1. **prediction**
        1. **target_predicton.csv**
        Результат прогнозирования бейзлайна (прогноз по срезу парка на 1 декабря 2022)
        2. **target_predicton_true.csv**
        Эталонные метки (для аналогичного периода)
    2. **target**
        1. **y_predict.csv**
        Номера вагонов и даты, по которым необходимо сделать предсказание. Приджойните к этому файлу ваши ответы и загрузите на платформу.
        2. **y_predict_submit_example.csv**
        Пример файла, который вы загрузите на платформу в качестве ответов.
        3. **y_train.csv**
        Данные для обучения.
    3. **metrics_f1.py**
    Скрипт расчёта скора
    4. **task2_base_model.ipynb**
    Baseline для быстрого старта
    
    [**Таблица с описанием файлов .parquet**](https://docs.google.com/spreadsheets/d/1SL2LZFZMnTY5q7iPi9GSG8Da2sWqPNR8/edit?usp=sharing&ouid=104910638716404157107&rtpof=true&sd=true)
    
    1. dislok_wagons.parquet
    2. freight_info.parquet
    3. kti_izm.parquet
    4. pr_rems.parquet
    5. stations.parquet
    6. tr_rems.parquet
    7. wag_params.parquet
    8. wagons_probeg_ownersip.parquet

1. **train_2.rar** - данные с 1 февраля по 28 февраля

<aside>
⚠️ **train_2.rar**
появится в папке с данными **12 ноября в 8:00 мск**

</aside>

### 💎 Описание итогового продукта

**Решение должно представлять собой:**

- Воспроизводимый код решения в .zip архиве с комментариями (даже если участник использовал сторонние ресурсы, код должен воспроизводиться корректно). Время и ресурсы во время тестирования ограничены. Внутрь архива также можно прикладывать файлы, необходимые для запуска решения (например, веса обученных моделей).
- В случае использования внешних файлов необходимо также приложить и указать ссылки/инструкции на скачивание.

---

### 🧑‍⚖️ **Критерии оценки решений**

**Победитель определяется по трём критериям, каждый из которых оценивается от 0 до 10 баллов:**

- **60% - Техническая реализация**
Скор в контесте

---

**Для того, чтобы ваш скор был учтен в рейтинге необходимо, чтобы:**
1. Решение было воспроизводимо(не забудьте зафискировать seed)
2. Код был “человекочитаемый” (избегайте длинных названий переменных, поддерживайте структуру кода)
3. В коде даны понятные комментарии, раскрывающие логику обработки данных и обучения моделей/работы алгоритма

Помните, что данные разбиты на паблик и приват.

---

- **30% - Бизнес-ценность**
Внедримо ли ваше решение в реальные бизнес-процессы?

---

Если в улучшении позиции в рейтинге вы застопорились, попробуйте проанализировать, какие признаки объединяют группы вагонов, отправленных в плановый ремонт. Поделитесь своими успешными исследованиями на презентации и в коде. Это будет учтено в итоговом рейтинге.

---

- **10% - Качество презентации**
Насколько ваша команда хорошо выступила перед жюри

---

**Общие рекомендации к презентации:**

1. Проанализировать важность фичей — интерпретировать их
2. Продумать механизм внедрения модели. Кто будет пользоваться и как?
3. Исследовать устойчивость модели: будет ли она работать на других данных? Как её тюнить? Предложить процесс.

---

### 📙 **Полезные материалы и идеи**

Ознакомьтесь со статьями на Хаб, чтобы вникнуть в контекст:
1. [Можно ли снизить затраты на ремонт вагонов?](https://habr.com/ru/companies/pgk/articles/695834/)
2. [ИИ в депо: 7 вопросов от ChatGPT про работу вагоноремонтного предприятия](https://habr.com/ru/companies/pgk/articles/738792/)
