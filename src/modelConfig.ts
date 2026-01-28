/**
 * Конфиг
 */


// Общие параметры датасета
/** Общее количество элементов в датасете */
export const NUM_DATASET_ELEMENTS = 100;

/** Количество элементов для обучения */
export const NUM_TRAIN_ELEMENTS = 80;

/** Количество элементов для тестирования */
export const NUM_TEST_ELEMENTS = 20;

/** Размер изображения (28x28 = 784 пикселей) */
export const IMAGE_SIZE = 784;

/** Количество классов (цифры от 0 до 9) */
export const NUM_CLASSES = 10;

/** Соотношение между обучающими и тестовыми данными */
export const TRAIN_TEST_RATIO = 4.0;



// Параметры обучения модели
/** Размер батча для обучения */
export const BATCH_SIZE = 64;

/** Количество эпох обучения */
export const EPOCHS = 5;

/** Скорость обучения (learning rate) для оптимизатора Adam */
export const LEARNING_RATE = 0.001;



// Архитектура модели
/** Размер ядра свертки */
export const KERNEL_SIZE = 3;

/** Количество фильтров в первом сверточном слое */
export const FILTERS_LAYER_1 = 8;

/** Количество фильтров во втором сверточном слое */
export const FILTERS_LAYER_2 = 16;

/** Размер пулинга (размер области для операции max pooling) */
export const POOL_SIZE = [2, 2];

/** Размеры входного изображения */
export const INPUT_SHAPE: [number, number, number] = [28, 28, 1];


// Параметры автоматической классификации
/** Интервал автоматической классификации в миллисекундах */
export const AUTO_CLASSIFICATION_INTERVAL = 5000;


// оптимизация
/** Размер чанка для загрузки данных */
export const CHUNK_SIZE = 5000;
