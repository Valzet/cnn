import * as tf from "@tensorflow/tfjs";
import { AUTO_CLASSIFICATION_INTERVAL } from "../modelConfig";
import type { DigitImageData } from "../ui/canvas";
import { displayDigitOnCanvas, generateRandomDigitImage, preprocessImage } from "../ui/canvas";
import { displayPredictionResults } from "../ui/predictionDisplay";

export function startAutoClassification(
  model: tf.LayersModel | undefined,
  digitCanvasCtx: CanvasRenderingContext2D | null
) {
  if (!model) {
    console.error("Model is not defined");
    return;
  }

  setInterval(async () => {
    if (!model) {
      console.error("Модель еще не готова");
      return;
    }

    const randomDigitImageData: DigitImageData = generateRandomDigitImage();
    const digitLabel = randomDigitImageData.digit;

    displayDigitOnCanvas(randomDigitImageData.imageData, digitLabel, digitCanvasCtx);

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

export function animateProcessingPipeline() {
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