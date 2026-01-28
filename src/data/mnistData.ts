import * as tf from "@tensorflow/tfjs";
import {
  NUM_DATASET_ELEMENTS,
  NUM_TRAIN_ELEMENTS,
  NUM_TEST_ELEMENTS,
  IMAGE_SIZE,
  NUM_CLASSES,
  TRAIN_TEST_RATIO,
  CHUNK_SIZE
} from "../modelConfig";
import { MNIST_IMAGES_SPRITE_PATH, MNIST_LABELS_PATH } from "../urls";

export class MnistData {
  dataset: Float32Array | null = null;
  labels: Uint8Array | null = null;
  trainIndices: Uint32Array | null = null;
  testIndices: Uint32Array | null = null;
  shuffledTestIndex = 0;
  shuffledTrainIndex = 0;

  async load() {
    const img = new Image();
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d")!;

    return new Promise<void>((resolve) => {
      img.crossOrigin = "";
      img.onload = () => {
        img.width = img.naturalWidth;
        img.height = img.naturalHeight;

        const datasetBytesBuffer = new ArrayBuffer(
          NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4,
        );

        const chunkSize = CHUNK_SIZE;
        canvas.width = img.width;
        canvas.height = img.height;

        for (let i = 0; i < NUM_DATASET_ELEMENTS; i += chunkSize) {
          const actualChunkSize = Math.min(chunkSize, NUM_DATASET_ELEMENTS - i);
          const datasetBytesView = new Float32Array(
            datasetBytesBuffer,
            i * IMAGE_SIZE * 4,
            actualChunkSize * IMAGE_SIZE,
          );

          const startY = i;
          const endY = Math.min(i + chunkSize, NUM_DATASET_ELEMENTS);
          const height = endY - startY;

          ctx.drawImage(
            img,
            0,
            startY,
            img.width,
            height,
            0,
            0,
            img.width,
            height,
          );

          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

          for (let j = 0; j < imageData.data.length / 4; j++) {
            datasetBytesView[j] = imageData.data[j * 4] / 255;
          }
        }

        this.dataset = new Float32Array(datasetBytesBuffer);

        fetch(MNIST_LABELS_PATH)
          .then((response) => response.arrayBuffer())
          .then((buffer) => new Uint8Array(buffer))
          .then((labels) => {
            this.labels = labels;

            this.trainIndices =
              this.createShuffledIndices(NUM_TRAIN_ELEMENTS);
            this.testIndices = this.createShuffledIndices(NUM_TEST_ELEMENTS);

            resolve();
          });
      };

      img.src = MNIST_IMAGES_SPRITE_PATH;
    });
  }

  createShuffledIndices(numIndices: number): Uint32Array {
    // Используем tf.util.createShuffledIndices если доступно
    if ((tf.util as any).createShuffledIndices) {
      return (tf.util as any).createShuffledIndices(numIndices);
    }

    // Альтернативная реализация если tf.util.createShuffledIndices недоступна
    const indices = new Uint32Array(numIndices);
    for (let i = 0; i < numIndices; i++) {
      indices[i] = i;
    }
    // Простая реализация перемешивания
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    return indices;
  }

  nextTrainBatch(batchSize: number) {
    return this.nextBatch(
      batchSize,
      [this.trainIndices!, this.shuffledTrainIndex],
      () => {
        this.shuffledTrainIndex += batchSize * TRAIN_TEST_RATIO;
      },
    );
  }

  nextTestBatch(batchSize: number) {
    return this.nextBatch(
      batchSize,
      [this.testIndices!, this.shuffledTestIndex],
      () => {
        this.shuffledTestIndex += batchSize;
      },
    );
  }

  nextBatch(
    batchSize: number,
    data: [Uint32Array, number],
    incrementFn: () => void,
  ) {
    const batchIndices = data[0].slice(data[1], data[1] + batchSize);
    incrementFn();

    const xsData = new Float32Array(batchSize * IMAGE_SIZE);
    const labelsData = new Uint8Array(batchSize * NUM_CLASSES);

    for (let i = 0; i < batchIndices.length; i++) {
      const idx = batchIndices[i];
      xsData.set(
        this.dataset!.subarray(idx * IMAGE_SIZE, (idx + 1) * IMAGE_SIZE),
        i * IMAGE_SIZE,
      );

      labelsData[i * NUM_CLASSES + this.labels![idx]] = 1;
    }

    const xs = tf.tensor2d(xsData, [batchSize, IMAGE_SIZE]);
    const labels = tf.tensor2d(labelsData, [batchSize, NUM_CLASSES]);

    return {
      xs: xs.reshape([batchSize, 28, 28, 1]),
      labels,
    };
  }
}