import os
import dill
import pandas as pd
import logging
from datetime import datetime
import json
from sklearn.pipeline import Pipeline

# Укажем путь к файлам проекта
path = os.environ.get('PROJECT_PATH', '.')
path_to_models = f'{path}/data/models'
path_to_test = f'{path}/data/test'
path_to_predictions = f'{path}/data/predictions'


def download_best_model() -> Pipeline:
    model_files = sorted(
        [f for f in os.listdir(path_to_models) if f.endswith('.pkl')], reverse=True
    )

    if not model_files:
        raise FileNotFoundError("Нет доступных моделей в data/models")

    best_model_path = os.path.join(path_to_models, model_files[0])

    with open(best_model_path, 'rb') as file:
        model = dill.load(file)

    logging.info(f'Загружена модель: {best_model_path}')
    return model


def load_test_data() -> pd.DataFrame:
    test_files = [f for f in os.listdir(path_to_test) if f.endswith('.json')]
    data_frames = []

    if not test_files:
        logging.warning('В папке data/test нет тестовых данных')
        return pd.DataFrame()  # Возвращаем пустой DataFrame

    for file in test_files:
        file_path = os.path.join(path_to_test, file)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            df = pd.DataFrame([data])
            data_frames.append(df)



        except ValueError as e:
            logging.error(f'Ошибка загрузки {file}: {e}')
            continue

    return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()


def predict():
    model = download_best_model()
    X_test = load_test_data()

    if X_test.empty:
        logging.warning('Предсказания не выполняются, так как тестовые данные отсутствуют')
        return

    preds = model.predict(X_test)
    X_test['prediction'] = preds

    os.makedirs(path_to_predictions, exist_ok=True)
    pred_filename = f'predictions_{datetime.now().strftime("%Y%m%d%H%M")}.csv'
    pred_path = os.path.join(path_to_predictions, pred_filename)
    X_test.to_csv(pred_path, index=False)

    logging.info(f'Предсказания сохранены в {pred_path}')


if __name__ == '__main__':
    predict()
