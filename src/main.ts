import "./style.css";
import * as tf from "@tensorflow/tfjs";
import {
  NUM_DATASET_ELEMENTS,
  NUM_TRAIN_ELEMENTS,
  NUM_TEST_ELEMENTS,
  IMAGE_SIZE,
  NUM_CLASSES,
  TRAIN_TEST_RATIO,
  BATCH_SIZE,
  EPOCHS,
  KERNEL_SIZE,
  FILTERS_LAYER_1,
  FILTERS_LAYER_2,
  POOL_SIZE,
  INPUT_SHAPE,
  AUTO_CLASSIFICATION_INTERVAL,
  CHUNK_SIZE,
} from "./modelConfig";
import { MNIST_IMAGES_SPRITE_PATH, MNIST_LABELS_PATH } from "./urls";

let model: tf.LayersModel;

let digitCanvasCtx: CanvasRenderingContext2D | null = null;

let predictionHistory: Array<{
  digit: number;
  predictedDigit: number;
  probabilities: number[];
  timestamp: Date;
}> = [];

tf.setBackend("cpu");
tf.ready().then(async () => {
  console.log("TensorFlow.js готов к работе");

  const digitCanvas = document.getElementById(
    "digit-canvas",
  ) as HTMLCanvasElement;
  if (digitCanvas) {
    digitCanvasCtx = digitCanvas.getContext("2d")!;

    if (digitCanvasCtx) {
      digitCanvasCtx.imageSmoothingEnabled = false;
    }
  }

  const data = new MnistData();
  await data.load();

  model = createModel();

  const modelSummaryContainer = document.getElementById(
    "model-summary-container",
  );
  if (modelSummaryContainer) {
    try {
      let layersHtml = "<h3>Архитектура модели:</h3>";
      layersHtml += '<div class="model-architecture">';

      model.layers.forEach((layer, index) => {
        const layerType = layer.getClassName();
        const paramsCount = layer.countParams();

        let layerClass = "layer-item";
        if (layerType.includes("Conv")) layerClass += " conv-layer";
        else if (layerType.includes("Pool")) layerClass += " pool-layer";
        else if (layerType.includes("Dense")) layerClass += " dense-layer";
        else if (layerType.includes("Flatten")) layerClass += " flatten-layer";

        layersHtml += `
          <div class="${layerClass}">
            <div class="layer-header">
              <span class="layer-index">Слой ${index + 1}</span>
              <span class="layer-name">${layer.name}</span>
            </div>
            <div class="layer-details">
              <div class="layer-type">${layerType}</div>
              <div class="layer-params">Параметров: ${paramsCount}</div>
            </div>
          </div>
        `;
      });

      layersHtml += "</div>";

      const totalParams = model.layers.reduce(
        (count, layer) => count + layer.countParams(),
        0,
      );
      const trainableParams = model.layers.reduce((count, layer) => {
        return count + (layer.trainable ? layer.countParams() : 0);
      }, 0);

      layersHtml += `
        <div class="model-stats">
          <div class="stat-item"><strong>Всего параметров:</strong> ${totalParams}</div>
          <div class="stat-item"><strong>Обучаемых параметров:</strong> ${trainableParams}</div>
        </div>
      `;

      modelSummaryContainer.innerHTML = layersHtml;
    } catch (error) {
      console.error("Ошибка при отображении архитектуры модели:", error);
      modelSummaryContainer.innerHTML =
        "<p>Не удалось отобразить архитектуру модели</p>";
    }
  }

  visualizeLayers(model);

  model.compile({
    optimizer: tf.train.adam(),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  const trainData = data.nextTrainBatch(200);
  const testData = data.nextTestBatch(100);

  let callbacks;
  let epochHistory: {
    epoch: number;
    loss: number;
    acc: number;
    val_loss: number;
    val_acc: number;
  }[] = [];

  callbacks = {
    onEpochEnd: async (epoch: number, logs: any) => {
      console.log(
        `Epoch ${epoch}: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.acc.toFixed(4)}`,
      );

      epochHistory.push({
        epoch,
        loss: logs.loss,
        acc: logs.acc,
        val_loss: logs.val_loss,
        val_acc: logs.val_acc,
      });

      const container = document.getElementById("training-history-container");
      if (container) {
        container.innerHTML = `
          <h3>История обучения:</h3>
          <table border="1" style="border-collapse: collapse;">
            <thead>
              <tr>
                <th>Эпоха</th>
                <th>Loss</th>
                <th>Accuracy</th>
                <th>Val Loss</th>
                <th>Val Accuracy</th>
              </tr>
            </thead>
            <tbody>
              ${epochHistory
                .map(
                  (log) =>
                    `<tr>
                  <td>${log.epoch}</td>
                  <td>${log.loss.toFixed(4)}</td>
                  <td>${log.acc.toFixed(4)}</td>
                  <td>${log.val_loss.toFixed(4)}</td>
                  <td>${log.val_acc.toFixed(4)}</td>
                </tr>`,
                )
                .join("")}
            </tbody>
          </table>
        `;
      }
    },
  };

  await model.fit(trainData.xs, trainData.labels, {
    batchSize: BATCH_SIZE,
    validationData: [testData.xs, testData.labels],
    epochs: EPOCHS,
    shuffle: true,
    callbacks,
  });

  const testResult = model.predict(testData.xs) as tf.Tensor;
  const axis = 1;
  const predictions = Array.from(testResult.argMax(axis).dataSync());
  const labels = Array.from(testData.labels.argMax(axis).dataSync());

  console.log("Матрица ошибок:", { labels, predictions });

  const predContainer = document.getElementById("prediction-results-container");
  if (predContainer) {
    predContainer.innerHTML = `
      <h3>Результаты предсказания:</h3>
      <p>Количество меток: ${labels.length}</p>
      <p>Количество предсказаний: ${predictions.length}</p>
      <p>Точность: ${((labels.filter((label, idx) => label === predictions[idx]).length / labels.length) * 100).toFixed(2)}%</p>
    `;
  }

  trainData.xs.dispose();
  trainData.labels.dispose();
  testData.xs.dispose();
  testData.labels.dispose();
  testResult.dispose();

  startAutoClassification();
});

function createModel(): tf.LayersModel {
  const model = tf.sequential({
    layers: [
      tf.layers.conv2d({
        inputShape: INPUT_SHAPE,
        kernelSize: KERNEL_SIZE,
        filters: FILTERS_LAYER_1,
        strides: 1,
        activation: "relu",
        kernelInitializer: "varianceScaling",
      }),

      tf.layers.maxPooling2d({
        poolSize: POOL_SIZE,
        strides: POOL_SIZE,
      }),

      tf.layers.conv2d({
        kernelSize: KERNEL_SIZE,
        filters: FILTERS_LAYER_2,
        strides: 1,
        activation: "relu",
        kernelInitializer: "varianceScaling",
      }),

      tf.layers.maxPooling2d({
        poolSize: POOL_SIZE,
        strides: POOL_SIZE,
      }),

      tf.layers.flatten(),

      tf.layers.dense({
        units: NUM_CLASSES,
        kernelInitializer: "varianceScaling",
        activation: "softmax",
      }),
    ],
  });

  return model;
}

function visualizeLayers(model: tf.LayersModel) {
  const layerInfo: { name: string; className: string; config: any }[] = [];
  model.layers.forEach((layer) => {
    layerInfo.push({
      name: layer.name,
      className: layer.getClassName(),
      config: layer.getConfig(),
    });
  });

  console.log("Информация о слоях:", layerInfo);
}

class MnistData {
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
              tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS);
            this.testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS);

            resolve();
          });
      };

      img.src = MNIST_IMAGES_SPRITE_PATH;
    });
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


function preprocessImage(imageData: ImageData): tf.Tensor4D {
  const pixels = tf.browser.fromPixels(imageData, 1);

  const resized = tf.image.resizeBilinear(pixels, [28, 28]);

  const normalized = resized.div(255.0);

  const batched = normalized.expandDims(0);

  return batched as tf.Tensor4D;
}

function displayPredictionResults(
  probabilities: number[],
  resultContainer: HTMLDivElement,
  autoClassification: boolean = false,
) {
  if (autoClassification) {
    resultContainer.innerHTML =
      "<h3>Результаты автоматической классификации:</h3>";
  } else {
    resultContainer.innerHTML = "<h3>Результаты классификации:</h3>";
  }

  const probabilityList = probabilities.map((prob, index) => ({
    digit: index,
    probability: prob,
  }));

  probabilityList.sort((a, b) => b.probability - a.probability);

  const topResults = probabilityList.slice(0, 5);

  const resultList = document.createElement("ul");
  topResults.forEach((item) => {
    const listItem = document.createElement("li");
    listItem.textContent = `Цифра ${item.digit}: ${(item.probability * 100).toFixed(2)}%`;
    resultList.appendChild(listItem);
  });

  resultContainer.appendChild(resultList);

  const digitLabelElement = document.getElementById("digit-label");
  let recognizedDigit = -1;
  if (digitLabelElement) {
    const text = digitLabelElement.textContent || "";
    const match = text.match(/Цифра: (\d)/);
    if (match) {
      recognizedDigit = parseInt(match[1]);
    }
  }

  const predictedDigit = probabilityList[0].digit;

  predictionHistory.push({
    digit: recognizedDigit,
    predictedDigit: predictedDigit,
    probabilities: [...probabilities],
    timestamp: new Date(),
  });

  if (predictionHistory.length > 40) {
    predictionHistory = predictionHistory.slice(-40);
  }

  const predContainer = document.getElementById("prediction-results-container");
  if (predContainer) {
    const recentPredictions = predictionHistory.slice(-10);
    predContainer.innerHTML = `
      <h3>Последние предсказания:</h3>
      <div class="recent-predictions">
        ${recentPredictions
          .map(
            (prediction) => `
          <div class="prediction-item">
            <div class="prediction-header">
              <span class="timestamp">${prediction.timestamp.toLocaleTimeString()}</span>
              <span class="digit-pair">Цифра: ${prediction.digit} → Предсказано: ${prediction.predictedDigit}</span>
            </div>
            <div class="prediction-match">${prediction.digit === prediction.predictedDigit ? "✓ Верно" : "✗ Ошибка"}</div>
          </div>
        `,
          )
          .join("")}
      </div>
    `;
  }

  updateProbabilityBars(probabilities);
}

function updateProbabilityBars(probabilities: number[]) {
  const probabilityBarsContainer = document.getElementById("probability-bars");
  if (!probabilityBarsContainer) return;

  probabilityBarsContainer.innerHTML = "";

  for (let i = 0; i < NUM_CLASSES; i++) {
    const probability = probabilities[i] || 0;
    const percentage = (probability * 100).toFixed(2);

    const barContainer = document.createElement("div");
    barContainer.className = "probability-bar-container";

    barContainer.innerHTML = `
      <div class="probability-label">${i}</div>
      <div class="probability-bar">
        <div class="probability-fill" style="width: ${percentage}%"></div>
        <span class="probability-percent">${percentage}%</span>
      </div>
    `;

    probabilityBarsContainer.appendChild(barContainer);
  }
}

function startAutoClassification() {
  setInterval(async () => {
    if (!model) {
      console.error("Модель еще не готова");
      return;
    }

    const randomDigitImageData = generateRandomDigitImage();
    const digitLabel = randomDigitImageData.digit;

    displayDigitOnCanvas(randomDigitImageData.imageData, digitLabel);

    animateProcessingPipeline();

    const tensor = preprocessImage(randomDigitImageData.imageData);

    const prediction = model.predict(tensor) as tf.Tensor;
    const probabilities = Array.from(await prediction.data());

    const resultContainer = document.getElementById(
      "result-container",
    ) as HTMLDivElement;
    if (resultContainer) {
      displayPredictionResults(probabilities, resultContainer, true);
    }

    tensor.dispose();
    prediction.dispose();
  }, AUTO_CLASSIFICATION_INTERVAL);
}

function animateProcessingPipeline() {
  const stages = document.querySelectorAll(".stage");

  stages.forEach((stage) => {
    stage.classList.remove("active", "processing");
  });

  stages.forEach((stage, index) => {
    setTimeout(() => {
      stage.classList.add("active", "processing");

      setTimeout(() => {
        stage.classList.remove("active", "processing");
      }, 1000);
    }, index * 300);
  });
}

function displayDigitOnCanvas(imageData: ImageData, digit: number) {
  if (!digitCanvasCtx) {
    console.error("Canvas context not initialized");
    return;
  }

  // Scale the 28x28 image to fit the canvas (140x140)
  const scale = 5;
  const scaledCanvas = document.createElement("canvas");
  const scaledCtx = scaledCanvas.getContext("2d")!;

  scaledCanvas.width = 28 * scale;
  scaledCanvas.height = 28 * scale;

  scaledCtx.putImageData(imageData, 0, 0);

  digitCanvasCtx.imageSmoothingEnabled = false;
  digitCanvasCtx.clearRect(
    0,
    0,
    digitCanvasCtx.canvas.width,
    digitCanvasCtx.canvas.height,
  );
  digitCanvasCtx.drawImage(
    scaledCanvas,
    0,
    0,
    28,
    28,
    0,
    0,
    28 * scale,
    28 * scale,
  );

  const digitLabelElement = document.getElementById("digit-label");
  if (digitLabelElement) {
    digitLabelElement.textContent = `Цифра: ${digit}`;
  }
}

type DigitImageData = {
  imageData: ImageData;
  digit: number;
};

function generateRandomDigitImage(): DigitImageData {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d")!;

  canvas.width = 28;
  canvas.height = 28;

  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const digit = Math.floor(Math.random() * NUM_CLASSES);

  ctx.font = "bold 20px Arial";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = "white";
  ctx.fillText(digit.toString(), canvas.width / 2, canvas.height / 2);

  return {
    imageData: ctx.getImageData(0, 0, canvas.width, canvas.height),
    digit: digit,
  };
}
