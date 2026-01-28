import "./style.css";
import * as tf from "@tensorflow/tfjs";
import {
  BATCH_SIZE,
  EPOCHS,
} from "./modelConfig";
import { MnistData } from "./data/mnistData";
import { createModel, visualizeLayers } from "./ml/model";
import { startAutoClassification } from "./core/autoClassification";

let model: tf.LayersModel;

let digitCanvasCtx: CanvasRenderingContext2D | null = null;

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

  startAutoClassification(model, digitCanvasCtx);
});





